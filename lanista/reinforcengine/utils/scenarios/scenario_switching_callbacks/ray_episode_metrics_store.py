# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed ownership of RLlib's framework-managed episode metric dictionary.

RLlib serializes ``episode.custom_data`` between workers, so the persisted
representation intentionally stays a plain dictionary.  This module is the
single place that creates that dictionary, exposes its typed view, and records
the quality of optional metric sources.
"""

from __future__ import annotations

import logging
from typing import Final, cast

from lanista.reinforcengine.rl_metrics.channel_schema import (
    DIAGNOSTIC_OBSERVATION_PARAM_NAMES,
    INTERACTION_EPISODE_METRIC_NAMES,
    POLICY_OBSERVATION_PARAM_NAMES,
    REWARD_SERIES_KEYS,
    TRAINING_DIAGNOSTIC_PARAM_NAMES,
)
from lanista.reinforcengine.rl_metrics.training_state_projection import (
    SCENARIO_ACTIVE_METRIC,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.ray_episode_metrics_contracts import (
    EpisodeDataPort,
    EpisodeMetricStore,
    MetricSeries,
    OptionalTelemetryQuality,
)

_LOGGER = logging.getLogger(__name__)

EPISODE_SCALAR_SERIES_KEYS: Final = (
    "episode_terminated",
    "episode_truncated",
    "episode_end_conflict",
    "episode_end_conflict_target_time_limit",
    "episode_end_conflict_target_red_light_overdrive",
    "curriculum_target_detected",
    "curriculum_target_accepted",
    "curriculum_target_reached",
    "curriculum_target_rejected_by_red_light_overdrive",
    "termination_curriculum_target",
    "termination_route_end",
    "truncation_time_limit",
    "truncation_red_light_overdrive",
    *INTERACTION_EPISODE_METRIC_NAMES,
    "terminated_reached_intersection",
    "terminated_max_speed",
    "terminated_average_speed",
    "truncated_reached_intersection",
    "truncated_max_speed",
    "truncated_average_speed",
    "max_speed_per_episode",
    "average_speed_per_episode",
    "reached_intersection_number",
    "number_of_vehicles",
    "number_of_pedestrians",
)

LEGACY_REWARD_METRIC_TO_SERIES_KEY: Final = {
    "sync_reward": "sync_raw",
    "normalized_sync_reward": "sync_normalized",
    "avg_sync_penalty": "sync_rolling_average",
    "accel_change_reward": "accel_raw",
    "normalized_accel_change_reward": "accel_normalized",
    "avg_accel_penalty": "accel_rolling_average",
    "avg_speed_reward": "speed_raw",
    "normalized_avg_speed_reward": "speed_normalized",
    "delta_change_reward": "delta_raw",
    "normalized_delta_change_reward": "delta_normalized",
    "avg_delta_penalty": "delta_rolling_average",
}

EPISODE_SETUP_METRICS: Final = {
    "warmup_seconds": "training_warmup_seconds",
    "warmup_steps": "training_warmup_steps",
    "warmup_max_seconds": "training_warmup_max_seconds",
    "warmup_max_steps": "training_warmup_max_steps",
    "startup_steps": "training_startup_steps",
    "active_policy_step_limit": "active_policy_step_limit",
    "active_policy_seconds_limit": "active_policy_seconds_limit",
}

CURRICULUM_METRICS: Final = (
    "active_level",
    "target_intersection",
    "required_terminated_episodes",
    "horizon_is_full",
)
SCENARIO_METRICS: Final = (SCENARIO_ACTIVE_METRIC,)


def build_episode_metric_store() -> EpisodeMetricStore:
    """Build the stable dictionary shape consumed by all metric stages."""

    data: dict[str, object] = {key: [] for key in EPISODE_SCALAR_SERIES_KEYS}
    data["curriculum_metrics"] = {key: [] for key in CURRICULUM_METRICS}
    data["scenario_metrics"] = {key: [] for key in SCENARIO_METRICS}
    data["telemetry_series"] = {
        name: [] for name in DIAGNOSTIC_OBSERVATION_PARAM_NAMES
    }
    data["diagnostic_series"] = {
        name: [] for name in TRAINING_DIAGNOSTIC_PARAM_NAMES
    }
    data["optional_telemetry_quality"] = {}
    data["optional_telemetry_degradation_causes"] = []
    data.update({key: [] for key in LEGACY_REWARD_METRIC_TO_SERIES_KEY})
    data["observation_series"] = {name: [] for name in POLICY_OBSERVATION_PARAM_NAMES}
    data["reward_series"] = {key: [] for key in REWARD_SERIES_KEYS}
    data["episode_setup_series"] = {key: [] for key in EPISODE_SETUP_METRICS}
    return cast(EpisodeMetricStore, data)


def episode_metric_store(episode: EpisodeDataPort) -> EpisodeMetricStore:
    """Return the typed view of the framework-owned ``lanistai`` dictionary."""

    return cast(EpisodeMetricStore, episode.custom_data["lanistai"])


def metric_series(store: EpisodeMetricStore, key: str) -> MetricSeries:
    """Return a dynamically selected scalar series from the typed store."""

    return cast(MetricSeries, cast(dict[str, object], store)[key])


def record_optional_quality(
    episode: EpisodeDataPort,
    *,
    source: str,
    quality: OptionalTelemetryQuality,
    cause: str | None = None,
    error: BaseException | None = None,
) -> None:
    """Append source fidelity and log each stable degradation only once."""

    data = cast(dict[str, object], episode_metric_store(episode))
    quality_series = cast(
        dict[str, list[float]],
        data.setdefault("optional_telemetry_quality", {}),
    )
    quality_series.setdefault(source, []).append(quality.score)
    if quality is OptionalTelemetryQuality.COMPLETE:
        return

    causes = cast(
        list[str],
        data.setdefault("optional_telemetry_degradation_causes", []),
    )
    stable_cause = cause or "unspecified"
    event_key = f"{source}:{stable_cause}"
    if event_key in causes:
        return
    causes.append(event_key)
    _LOGGER.warning(
        "Optional Ray telemetry degraded.",
        extra={
            "telemetry_source": source,
            "telemetry_quality": quality.value,
            "telemetry_cause": stable_cause,
            "exception_type": type(error).__name__ if error is not None else None,
        },
    )
