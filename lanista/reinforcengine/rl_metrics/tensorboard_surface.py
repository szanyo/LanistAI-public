# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Stable TensorBoard and dashboard projection policy for LanistAI metrics.

The metric schema describes values produced by environments and framework
adapters.  This module owns the separate decision of which of those values are
published, how they are named in TensorBoard, and which compact aliases feed
the dashboard.  Keeping the policy here makes tag changes auditable without
mixing them with reward or observation calculations.

All iterators preserve source insertion order.  When two flat metrics map to
the same output tag, the later metric replaces the value without moving that
tag, matching normal Python dictionary semantics and the historical behavior.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import numpy as np
from numpy.typing import NDArray

from lanista.reinforcengine.rl_metrics.aggregation import (
    flatten_numeric_metrics,
    is_valid_number,
)
from lanista.reinforcengine.rl_metrics.channel_schema import (
    INTERACTION_EPISODE_METRIC_NAMES,
    TRAINING_DIAGNOSTIC_PARAM_NAMES,
)
from lanista.reinforcengine.rl_metrics.tensorboard_schema import (
    COMPACT_SCALAR_METRIC_KEYS,
    COMPARABLE_CUMULATIVE_KPI_TAGS,
    COMPARABLE_KPI_TAGS,
    DIAGNOSTIC_HISTOGRAM_METRIC_KEYS,
    NUMBERED_DASHBOARD_KPI_TAGS,
    TENSORBOARD_HISTOGRAM_SAMPLE_LIMIT,
    TensorboardProfile,
    resolve_tensorboard_profile,
    training_histogram_tag,
    training_tensorboard_path,
    training_tensorboard_tag,
)

LANISTAI_PREFIX = ("lanistai",)

INFERENCE_ACTION_TAG = "lanistai_eval_motion/policy_action"
INFERENCE_REWARD_TAG = "lanistai_eval_reward/total"
INFERENCE_OBSERVATION_TAG = "lanistai_distribution/evaluation/observation"

EVALUATION_EPISODE_RETURN_TAG = "lanistai_eval_overview/episode_return"
EVALUATION_EPISODE_LENGTH_TAG = "lanistai_eval_overview/episode_length"

SCENARIO_ACTIVE_PATH = ("lanistai_overview", "active_scenario")
SCENARIO_ACTIVE_TAG = "/".join(SCENARIO_ACTIVE_PATH)

# Kept as a public compatibility name for the inspection dashboard.  Schema-v2
# writes these aliases as the only canonical overview tags, never as a second
# persisted copy of another metric.
DASHBOARD_METRIC_ALIASES: dict[str, str] = {
    **NUMBERED_DASHBOARD_KPI_TAGS,
    **{
        key: alias
        for key, alias in COMPARABLE_KPI_TAGS.items()
        if key not in NUMBERED_DASHBOARD_KPI_TAGS
    },
}

TENSORBOARD_DETAIL_STAT_NAMES = (
    "min",
    "max",
    "std",
    "p25",
    "p50",
    "p75",
    "range",
    "iqr",
)
TENSORBOARD_SCALAR_DETAIL_STAT_NAMES = (
    "min",
    "max",
    "std",
    "p25",
    "p50",
    "p75",
)
TENSORBOARD_SCALAR_METRIC_KEYS = frozenset(
    {
        *(
            f"diagnostics.{name}.stats.mean"
            for name in TRAINING_DIAGNOSTIC_PARAM_NAMES
        ),
        *(f"episode_info.{name}.mean" for name in INTERACTION_EPISODE_METRIC_NAMES),
        "episode_info.episode_terminated.mean",
        "episode_info.episode_terminated.sum",
        "episode_info.episode_truncated.mean",
        "episode_info.episode_truncated.sum",
        "episode_info.termination_curriculum_target.mean",
        "episode_info.termination_curriculum_target.sum",
        "episode_info.termination_route_end.mean",
        "episode_info.termination_route_end.sum",
        "episode_info.truncation_red_light_overdrive.mean",
        "episode_info.truncation_red_light_overdrive.sum",
        "episode_info.truncation_time_limit.mean",
        "episode_info.truncation_time_limit.sum",
        "episode_info.curriculum_target_detected.mean",
        "episode_info.curriculum_target_detected.sum",
        "episode_info.curriculum_target_accepted.mean",
        "episode_info.curriculum_target_accepted.sum",
        "episode_info.curriculum_target_reached.mean",
        "episode_info.curriculum_target_reached.sum",
        "episode_info.curriculum_target_rejected_by_red_light_overdrive.mean",
        "episode_info.curriculum_target_rejected_by_red_light_overdrive.sum",
        "episode_info.episode_end_conflict.mean",
        "episode_info.episode_end_conflict.sum",
        "episode_info.episode_end_conflict_target_time_limit.mean",
        "episode_info.episode_end_conflict_target_time_limit.sum",
        "episode_info.episode_end_conflict_target_red_light_overdrive.mean",
        "episode_info.episode_end_conflict_target_red_light_overdrive.sum",
        "episode_info.terminated_reached_intersection.mean",
        "episode_info.terminated_max_speed.mean",
        "episode_info.terminated_average_speed.mean",
        "episode_info.truncated_reached_intersection.mean",
        "episode_info.truncated_max_speed.mean",
        "episode_info.truncated_average_speed.mean",
        "episode_setup.warmup_seconds.value",
        "episode_setup.warmup_steps.value",
        "episode_setup.active_policy_step_limit.value",
        "episode_setup.active_policy_seconds_limit.value",
        "vehicle_info.average_speed_per_episode.mean",
        "vehicle_info.max_speed_per_episode.mean",
        "vehicle_info.reached_intersection_number.mean",
        "vehicle_info.number_of_vehicles.mean",
        "vehicle_info.number_of_pedestrians.mean",
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
        "observations.tl_phase_remaining_delta_norm.stats.mean",
        "observations.tl_phase_remaining_raw_delta_norm.stats.mean",
        "observations.tl_timing_context_changed.stats.mean",
        "observations.tl_phase_remaining_monotonic_violation.stats.mean",
        "observations.u1_applied_command.stats.mean",
        "observations.u1_actual_acceleration.stats.mean",
        "observations.u1_supervised_command.stats.mean",
        "observations.upcoming_intersection_type.stats.mean",
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
        "rewards.speed.signal_stop_observed_stats.mean",
        "rewards.speed.signal_stop_quality_stats.mean",
        "rewards.speed.signal_stop_positive_accel_penalty_stats.mean",
        "rewards.speed.signal_release_reward_stats.mean",
        "rewards.speed.pre_jitter_discount_effective_stats.mean",
        "rewards.speed.jitter_discount_stats.mean",
        "rewards.speed.jitter_discount_applied_stats.mean",
        "rewards.speed.weighted_stats.mean",
        "rewards.sync.weighted_stats.mean",
        "rewards.sync.command_stats.mean",
        "rewards.sync.actual_acceleration_stats.mean",
        "rewards.sync.absolute_error_stats.mean",
        "rewards.sync.sumo_intervention_stats.mean",
        "rewards.sync.lead_present_absolute_error_stats.mean",
        "rewards.sync.lead_absent_absolute_error_stats.mean",
        "rewards.sync.lead_present_sumo_intervention_stats.mean",
        "rewards.sync.lead_absent_sumo_intervention_stats.mean",
        "rewards.sync.effective_error_stats.mean",
        "rewards.sync.speed_factor_stats.mean",
        "rewards.sync.direction_penalty_stats.mean",
        "rewards.sync.magnitude_penalty_stats.mean",
        "rewards.delta.weighted_stats.mean",
        "rewards.accel.weighted_stats.mean",
        "rewards.control.policy_action_stats.mean",
        "rewards.control.supervised_cmd_accel_local_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_local_jitter_sign_change_stats.mean",
        "rewards.control.supervised_cmd_accel_local_effective_jitter_stats.mean",
        "rewards.control.lead_present_local_jitter_stats.mean",
        "rewards.control.lead_absent_local_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_trend_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_trend_jitter_sign_change_stats.mean",
        "rewards.control.supervised_cmd_accel_trend_effective_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_weighted_comfort_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_comfort_jitter_sign_change_stats.mean",
        "rewards.control.supervised_cmd_accel_comfort_effective_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_discount_jitter_stats.mean",
        "rewards.control.normalized_supervised_cmd_accel_discount_jitter_stats.mean",
        "rewards.control.abs_supervised_cmd_accel_delta_change_stats.mean",
    }
)

TENSORBOARD_HISTOGRAM_METRIC_KEYS = frozenset(
    {
        "observations.ego_speed_norm.stats.mean",
        "observations.u1_applied_command.stats.mean",
        "observations.u1_actual_acceleration.stats.mean",
        "observations.u1_supervised_command.stats.mean",
        "observations.red_light_distance_norm.stats.mean",
        "observations.red_light_preferred_speed_norm.stats.mean",
        "observations.red_light_brake_limit_speed_norm.stats.mean",
        "observations.red_brake_horizon_active.stats.mean",
        "observations.speed_error_to_preferred_norm.stats.mean",
        "observations.speed_error_to_brake_limit_norm.stats.mean",
        "observations.tl_phase_remaining_norm.stats.mean",
        "rewards.total.raw_stats.mean",
        "rewards.speed.effective_stats.mean",
        "rewards.speed.red_light_speed_reward_stats.mean",
        "rewards.speed.red_light_profile_reward_stats.mean",
        "rewards.speed.red_light_stopline_factor_stats.mean",
        "rewards.speed.red_light_far_wait_penalty_stats.mean",
        "rewards.speed.green_light_idle_penalty_stats.mean",
        "rewards.speed.blocking_profile_reward_stats.mean",
        "rewards.speed.signal_stop_quality_stats.mean",
        "rewards.speed.signal_stop_positive_accel_penalty_stats.mean",
        "rewards.speed.signal_release_reward_stats.mean",
        "rewards.speed.pre_jitter_discount_effective_stats.mean",
        "rewards.speed.jitter_discount_stats.mean",
        "rewards.speed.jitter_discount_applied_stats.mean",
        "rewards.sync.weighted_stats.mean",
        "rewards.sync.absolute_error_stats.mean",
        "rewards.sync.lead_present_absolute_error_stats.mean",
        "rewards.sync.lead_absent_absolute_error_stats.mean",
        "rewards.sync.effective_error_stats.mean",
        "rewards.delta.weighted_stats.mean",
        "rewards.accel.weighted_stats.mean",
        "rewards.control.policy_action_stats.mean",
        "rewards.control.supervised_cmd_accel_local_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_local_jitter_sign_change_stats.mean",
        "rewards.control.supervised_cmd_accel_local_effective_jitter_stats.mean",
        "rewards.control.lead_present_local_jitter_stats.mean",
        "rewards.control.lead_absent_local_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_trend_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_trend_jitter_sign_change_stats.mean",
        "rewards.control.supervised_cmd_accel_trend_effective_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_weighted_comfort_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_comfort_jitter_sign_change_stats.mean",
        "rewards.control.supervised_cmd_accel_comfort_effective_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_discount_jitter_stats.mean",
        "rewards.control.normalized_supervised_cmd_accel_discount_jitter_stats.mean",
    }
)

TENSORBOARD_DETAIL_SCALAR_METRIC_KEYS = frozenset(
    {
        "observations.ego_speed_norm.stats.mean",
        "observations.u1_applied_command.stats.mean",
        "observations.u1_actual_acceleration.stats.mean",
        "observations.u1_supervised_command.stats.mean",
        "observations.red_light_distance_norm.stats.mean",
        "observations.red_light_preferred_speed_norm.stats.mean",
        "observations.red_light_brake_limit_speed_norm.stats.mean",
        "observations.red_brake_horizon_active.stats.mean",
        "observations.speed_error_to_preferred_norm.stats.mean",
        "observations.speed_error_to_brake_limit_norm.stats.mean",
        "observations.tl_phase_remaining_norm.stats.mean",
        "rewards.total.raw_stats.mean",
        "rewards.speed.effective_stats.mean",
        "rewards.speed.red_light_speed_reward_stats.mean",
        "rewards.speed.red_light_profile_reward_stats.mean",
        "rewards.speed.red_light_stopline_factor_stats.mean",
        "rewards.speed.blocking_profile_reward_stats.mean",
        "rewards.speed.signal_stop_quality_stats.mean",
        "rewards.speed.signal_stop_positive_accel_penalty_stats.mean",
        "rewards.speed.signal_release_reward_stats.mean",
        "rewards.speed.pre_jitter_discount_effective_stats.mean",
        "rewards.speed.jitter_discount_stats.mean",
        "rewards.speed.jitter_discount_applied_stats.mean",
        "rewards.sync.weighted_stats.mean",
        "rewards.sync.absolute_error_stats.mean",
        "rewards.sync.effective_error_stats.mean",
        "rewards.delta.weighted_stats.mean",
        "rewards.accel.weighted_stats.mean",
        "rewards.control.policy_action_stats.mean",
        "rewards.control.supervised_cmd_accel_local_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_local_jitter_sign_change_stats.mean",
        "rewards.control.supervised_cmd_accel_local_effective_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_trend_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_trend_jitter_sign_change_stats.mean",
        "rewards.control.supervised_cmd_accel_trend_effective_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_weighted_comfort_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_comfort_jitter_sign_change_stats.mean",
        "rewards.control.supervised_cmd_accel_comfort_effective_jitter_stats.mean",
        "rewards.control.supervised_cmd_accel_discount_jitter_stats.mean",
        "rewards.control.normalized_supervised_cmd_accel_discount_jitter_stats.mean",
        "rewards.control.abs_supervised_cmd_accel_delta_change_stats.mean",
    }
)


def dashboard_metric_alias(flat_metric_key: str) -> str | None:
    """Return the stable compact dashboard alias for a flat metric key."""

    return DASHBOARD_METRIC_ALIASES.get(str(flat_metric_key))


def metric_path(*parts: str) -> tuple[str, ...]:
    """Build a tuple path rooted at the shared ``lanistai`` namespace."""

    return LANISTAI_PREFIX + tuple(str(part) for part in parts)


def dashboard_tensorboard_tag(alias: str) -> str:
    """Build the TensorBoard tag used by one dashboard alias."""

    return "/".join(("lanistai_overview", str(alias)))


def dashboard_metric_path(alias: str) -> tuple[str, ...]:
    """Build the tuple metric path used by RLlib's metrics logger."""

    return ("lanistai_overview", str(alias))


def dashboard_metric_values_from_flat_metrics(
    metrics: Mapping[str, float],
) -> dict[str, float]:
    """Project finite flat metrics onto the compact dashboard namespace."""

    values: dict[str, float] = {}
    for key, value in metrics.items():
        alias = dashboard_metric_alias(key)
        if alias is not None and is_valid_number(value):
            values[alias] = float(value)
    return values


def tensorboard_tag_from_flat_key(flat_metric_key: str) -> str:
    """Translate a dotted flat key into its canonical scalar tag."""

    return training_tensorboard_tag(flat_metric_key)


def tensorboard_metric_path_from_flat_key(
    flat_metric_key: str,
) -> tuple[str, ...]:
    """Translate a dotted flat key into its canonical RLlib tuple path."""

    return training_tensorboard_path(flat_metric_key)


def _schema_v2_scalar_metric_keys() -> tuple[str, ...]:
    """Return every canonical scalar key that schema-v2 may publish."""

    keys = set(TENSORBOARD_SCALAR_METRIC_KEYS)
    for mean_key in TENSORBOARD_DETAIL_SCALAR_METRIC_KEYS:
        if not mean_key.endswith(".mean"):
            continue
        stem = mean_key.removesuffix(".mean")
        keys.update(
            f"{stem}.{stat_name}"
            for stat_name in TENSORBOARD_DETAIL_STAT_NAMES
        )
    return tuple(sorted(keys))


_SCHEMA_V2_SCALAR_METRIC_KEYS = _schema_v2_scalar_metric_keys()


def _nested_metric_value(
    metrics: Mapping[str, object],
    path: tuple[str, ...],
) -> object | None:
    """Read one schema-v2 tuple path without fabricating missing values."""

    value: object = metrics
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def canonical_training_metrics_from_result(
    env_runners: Mapping[str, object],
) -> dict[str, float]:
    """Restore canonical flat metrics from legacy or schema-v2 Ray results."""

    legacy = env_runners.get("lanistai")
    if isinstance(legacy, dict):
        flattened = flatten_numeric_metrics(legacy)
        if flattened:
            return flattened

    restored: dict[str, float] = {}
    for flat_key in _SCHEMA_V2_SCALAR_METRIC_KEYS:
        value = _nested_metric_value(
            env_runners,
            training_tensorboard_path(flat_key),
        )
        if isinstance(value, bool) or not is_valid_number(value):
            continue
        restored[flat_key] = float(value)

    curriculum = env_runners.get("lanistai_curriculum")
    if isinstance(curriculum, dict):
        for key, value in flatten_numeric_metrics(curriculum).items():
            restored[f"curriculum.{key}"] = value
    return restored


def tensorboard_histogram_tag_from_flat_key(flat_metric_key: str) -> str:
    """Translate a scalar-stat key into its stable histogram tag."""

    return training_histogram_tag(flat_metric_key)


def _mean_key_for_detail_stat(flat_metric_key: str) -> str | None:
    """Return the mean key for detail stat value owned by this tensorboard surface module."""
    key = str(flat_metric_key)
    for stat_name in TENSORBOARD_DETAIL_STAT_NAMES:
        suffix = f".{stat_name}"
        if key.endswith(suffix):
            return f"{key[: -len(suffix)]}.mean"
    return None


def should_log_lanistai_tensorboard_scalar(
    flat_metric_key: str,
    *,
    include_stat_details: bool = False,
    profile: TensorboardProfile | str | None = None,
) -> bool:
    """Return whether a flat key belongs to the stable scalar surface."""

    key = str(flat_metric_key)
    resolved_profile = resolve_tensorboard_profile(profile)
    primary_keys = (
        TENSORBOARD_SCALAR_METRIC_KEYS
        if resolved_profile is TensorboardProfile.FULL
        else COMPACT_SCALAR_METRIC_KEYS
    )
    if key in primary_keys:
        return True
    detail_profile = resolved_profile
    if include_stat_details and detail_profile is TensorboardProfile.COMPACT:
        detail_profile = TensorboardProfile.DIAGNOSTIC
    if detail_profile is not TensorboardProfile.COMPACT:
        mean_key = _mean_key_for_detail_stat(key)
        stat_name = key.rsplit(".", 1)[-1]
        allowed_stats = (
            TENSORBOARD_DETAIL_STAT_NAMES
            if detail_profile is TensorboardProfile.FULL
            else ("min", "max", "std")
        )
        allowed_means = (
            TENSORBOARD_DETAIL_SCALAR_METRIC_KEYS
            if detail_profile is TensorboardProfile.FULL
            else COMPACT_SCALAR_METRIC_KEYS
        )
        return (
            stat_name in allowed_stats
            and mean_key in allowed_means
            and mean_key in TENSORBOARD_DETAIL_SCALAR_METRIC_KEYS
        )
    return False


def should_log_lanistai_tensorboard_histogram(
    flat_metric_key: str,
    *,
    profile: TensorboardProfile | str | None = None,
) -> bool:
    """Return whether a flat key belongs to the stable histogram surface."""

    resolved_profile = resolve_tensorboard_profile(profile)
    if resolved_profile is TensorboardProfile.COMPACT:
        return False
    allowed_keys = (
        TENSORBOARD_HISTOGRAM_METRIC_KEYS
        if resolved_profile is TensorboardProfile.FULL
        else DIAGNOSTIC_HISTOGRAM_METRIC_KEYS
    )
    return str(flat_metric_key) in allowed_keys


def tensorboard_histogram_values(
    values: Iterable[float],
    *,
    sample_limit: int = TENSORBOARD_HISTOGRAM_SAMPLE_LIMIT,
) -> NDArray[np.float64] | None:
    """Normalize finite histogram values and sample them deterministically."""

    try:
        arr = np.asarray(list(values), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    limit = max(1, int(sample_limit))
    if arr.size > limit:
        sample_indexes = np.linspace(0, arr.size - 1, num=limit, dtype=int)
        arr = arr[sample_indexes]
    return cast(NDArray[np.float64], arr.astype(float, copy=False))


def iter_lanistai_tensorboard_histogram_items(
    metric_values: Mapping[str, Iterable[float]],
    *,
    sample_limit: int = TENSORBOARD_HISTOGRAM_SAMPLE_LIMIT,
    profile: TensorboardProfile | str | None = None,
) -> Iterable[tuple[str, NDArray[np.float64]]]:
    """Yield ordered, allowlisted TensorBoard histogram tag/value pairs."""

    emitted: dict[str, NDArray[np.float64]] = {}
    for key, values in metric_values.items():
        if not should_log_lanistai_tensorboard_histogram(key, profile=profile):
            continue
        arr = tensorboard_histogram_values(values, sample_limit=sample_limit)
        if arr is None:
            continue
        emitted[tensorboard_histogram_tag_from_flat_key(key)] = arr
    return emitted.items()


def iter_lanistai_tensorboard_scalar_items(
    metrics: dict[str, object],
    *,
    include_stat_details: bool = False,
    profile: TensorboardProfile | str | None = None,
) -> Iterable[tuple[str, float]]:
    """Yield ordered, finite and allowlisted TensorBoard scalar pairs."""

    emitted: dict[str, float] = {}
    flat_metrics = flatten_numeric_metrics(metrics)
    for key, value in flat_metrics.items():
        if not is_valid_number(value):
            continue
        if not should_log_lanistai_tensorboard_scalar(
            key,
            include_stat_details=include_stat_details,
            profile=profile,
        ):
            continue
        emitted[tensorboard_tag_from_flat_key(key)] = float(value)
    return emitted.items()


def tensorboard_scalar_value(value: object) -> float:
    """Return the first scalar-like value, or zero for an empty value."""

    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])
