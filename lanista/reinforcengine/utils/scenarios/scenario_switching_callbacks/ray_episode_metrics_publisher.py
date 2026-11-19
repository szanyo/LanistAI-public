# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral TensorBoard publication for Ray episode metrics.

The publisher owns stable paths, reductions, and histogram selection.
It deliberately propagates mandatory logger failures while optional histogram
container failures remain observable through telemetry-quality metrics.
"""

from __future__ import annotations

from typing import cast

from lanista.reinforcengine.rl_metrics.aggregation import compute_stats, is_valid_number
from lanista.reinforcengine.rl_metrics.channel_schema import (
    DIAGNOSTIC_OBSERVATION_PARAM_NAMES,
    PER_EPISODE_METRICS,
    REWARD_SERIES_KEYS,
)
from lanista.reinforcengine.rl_metrics.tensorboard_schema import (
    TENSORBOARD_HISTOGRAM_CADENCE,
    TensorboardProfile,
    resolve_tensorboard_profile,
)
from lanista.reinforcengine.rl_metrics.tensorboard_surface import (
    SCENARIO_ACTIVE_PATH,
    iter_lanistai_tensorboard_histogram_items,
    should_log_lanistai_tensorboard_scalar,
    tensorboard_metric_path_from_flat_key,
)
from lanista.reinforcengine.rl_metrics.training_state_projection import (
    SCENARIO_ACTIVE_METRIC,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.ray_episode_metrics_contracts import (
    EpisodeDataPort,
    EpisodeMetricStore,
    FloatSeriesMap,
    MetricsLoggerPort,
    OptionalTelemetryQuality,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.ray_episode_metrics_store import (
    episode_metric_store,
    metric_series,
    record_optional_quality,
)


def _attribute(source: object, name: str, default: object = None) -> object:
    """Return the attribute value owned by this ray episode metrics publisher module."""
    return cast(object, getattr(source, name, default))


class EpisodeMetricsPublisher:
    """Publish captured values through a narrow tuple-path logger port."""

    def __init__(self) -> None:
        """Resolve one profile consistently for the publisher lifetime."""

        self._tensorboard_profile = resolve_tensorboard_profile()
        self._histogram_window_count = 0

    @staticmethod
    def initialize_environment_metrics(metrics_logger: MetricsLoggerPort) -> None:
        """Avoid fabricating zero metrics before an episode denominator exists."""

        _ = metrics_logger

    @staticmethod
    def _publish_scalar(
        metrics_logger: MetricsLoggerPort,
        *,
        flat_key: str,
        value: object,
        reduce_mode: str,
    ) -> None:
        """Publish one scalar through its sole schema-v2 tuple path."""

        metrics_logger.log_value(
            tensorboard_metric_path_from_flat_key(flat_key),
            value,
            reduce=reduce_mode,
        )

    def publish_episode_scalars(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish finite per-episode values in historical schema order."""

        for index in range(len(store["episode_terminated"])):
            for list_key, path_parts, reduce_modes in PER_EPISODE_METRICS:
                values = metric_series(store, list_key)
                if index >= len(values):
                    continue
                value = values[index]
                if not is_valid_number(value):
                    continue
                for reduce_mode in reduce_modes:
                    flat_key = ".".join(path_parts + (reduce_mode,))
                    if should_log_lanistai_tensorboard_scalar(
                        flat_key,
                        profile=self._tensorboard_profile,
                    ):
                        self._publish_scalar(
                            metrics_logger,
                            flat_key=flat_key,
                            value=value,
                            reduce_mode=reduce_mode,
                        )

    @staticmethod
    def publish_curriculum_metrics(
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish curriculum projections without mutating curriculum state."""

        for key, values in store["curriculum_metrics"].items():
            for value in values:
                if is_valid_number(value):
                    metrics_logger.log_value(
                        ("lanistai_curriculum", key),
                        float(value),
                        reduce="max",
                    )

    @staticmethod
    def publish_scenario_metrics(
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish shared scenario projections without mutating scenario state."""

        for value in store["scenario_metrics"][SCENARIO_ACTIVE_METRIC]:
            if is_valid_number(value):
                metrics_logger.log_value(
                    SCENARIO_ACTIVE_PATH,
                    float(value),
                    reduce="max",
                )

    @staticmethod
    def publish_intersection_metrics(
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish episode mean and maximum for finite reached intersections."""

        values = [
            float(value)
            for value in store["reached_intersection_number"]
            if is_valid_number(value)
        ]
        if not values:
            return
        metrics_logger.log_value(
            (
                "lanistai_vehicle",
                "reached_intersection_number",
                "episode_max",
            ),
            float(max(values)),
            reduce="max",
        )

    def publish_observation_statistics(
        self,
        series: FloatSeriesMap,
        metrics_logger: MetricsLoggerPort,
        *,
        diagnostic_only: bool,
    ) -> None:
        """Publish compact statistics without mixing diagnostic policy channels."""

        for name, values in series.items():
            if diagnostic_only and name not in DIAGNOSTIC_OBSERVATION_PARAM_NAMES:
                continue
            self._publish_series_statistics(
                metrics_logger,
                metric_parts=("observations", name, "stats"),
                values=values,
            )

    def _publish_series_statistics(
        self,
        metrics_logger: MetricsLoggerPort,
        *,
        metric_parts: tuple[str, ...],
        values: list[float],
    ) -> None:
        """Publish one finite statistic per episode.

        SB3 stores one computed statistic in each episode snapshot. Feeding
        every Ray step separately into MetricsLogger instead invokes Ray's
        default 0.01 EMA over steps and lets long episodes dominate the
        framework comparison. Compute the same episode statistic here, then
        let Ray aggregate those episode values.
        """

        stats = compute_stats(values)
        if stats is None:
            return
        for stat_name, stat_value in stats.items():
            flat_key = ".".join(metric_parts + (stat_name,))
            if should_log_lanistai_tensorboard_scalar(
                flat_key,
                profile=self._tensorboard_profile,
            ):
                reduction = stat_name if stat_name in {"min", "max"} else "mean"
                if is_valid_number(stat_value):
                    self._publish_scalar(
                        metrics_logger,
                        flat_key=flat_key,
                        value=float(stat_value),
                        reduce_mode=reduction,
                    )

    def publish_episode_setup_metrics(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish finite setup values with the historical mean reduction."""

        for key, values in store["episode_setup_series"].items():
            flat_key = f"episode_setup.{key}.value"
            if not should_log_lanistai_tensorboard_scalar(
                flat_key,
                profile=self._tensorboard_profile,
            ):
                continue
            for value in values:
                if is_valid_number(value):
                    metrics_logger.log_value(
                        tensorboard_metric_path_from_flat_key(flat_key),
                        float(value),
                        reduce="mean",
                    )

    def publish_reward_statistics(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish compact statistics for registered reward-component series."""

        for key, values in store["reward_series"].items():
            metric_parts = REWARD_SERIES_KEYS.get(key)
            if metric_parts is not None:
                self._publish_series_statistics(
                    metrics_logger,
                    metric_parts=metric_parts,
                    values=values,
                )

    def publish_diagnostic_statistics(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish semantic diagnostics through the episode aggregation window."""

        for name, values in store["diagnostic_series"].items():
            self._publish_series_statistics(
                metrics_logger,
                metric_parts=("diagnostics", name, "stats"),
                values=values,
            )

    @staticmethod
    def publish_optional_telemetry_quality(
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish source fidelity independently from domain metric defaults."""

        for source, values in store["optional_telemetry_quality"].items():
            for value in values:
                metrics_logger.log_value(
                    (
                        "lanistai_quality",
                        "optional_telemetry_quality",
                        source,
                        "score",
                    ),
                    float(value),
                    reduce="mean",
                )

    @staticmethod
    def _episode_hist_data(episode: EpisodeDataPort) -> dict[str, list[float]] | None:
        """Return the episode hist data value owned by this episode metrics publisher."""
        try:
            hist_data = _attribute(episode, "hist_data")
        except Exception as error:
            record_optional_quality(
                episode,
                source="histogram",
                quality=OptionalTelemetryQuality.UNAVAILABLE,
                cause="histogram_read_failed",
                error=error,
            )
            return None
        if isinstance(hist_data, dict):
            return cast(dict[str, list[float]], hist_data)
        try:
            object.__setattr__(episode, "hist_data", {})
            hist_data = _attribute(episode, "hist_data")
        except Exception as error:
            record_optional_quality(
                episode,
                source="histogram",
                quality=OptionalTelemetryQuality.UNAVAILABLE,
                cause="histogram_assignment_failed",
                error=error,
            )
            return None
        record_optional_quality(
            episode,
            source="histogram",
            quality=OptionalTelemetryQuality.FALLBACK,
            cause="histogram_container_created",
        )
        return cast(dict[str, list[float]], hist_data)

    def record_episode_histograms(self, episode: EpisodeDataPort) -> None:
        """Append allowlisted observation, telemetry, and reward histograms."""

        self._histogram_window_count += 1
        cadence = (
            1
            if self._tensorboard_profile is TensorboardProfile.FULL
            else TENSORBOARD_HISTOGRAM_CADENCE
        )
        if (
            self._tensorboard_profile is TensorboardProfile.COMPACT
            or self._histogram_window_count % cadence
        ):
            return

        store = episode_metric_store(episode)
        metric_values: dict[str, list[float]] = {
            f"observations.{name}.stats.mean": values
            for name, values in store["observation_series"].items()
        }
        for name, values in store["telemetry_series"].items():
            if name in DIAGNOSTIC_OBSERVATION_PARAM_NAMES:
                metric_values.setdefault(f"observations.{name}.stats.mean", values)
        for key, values in store["reward_series"].items():
            metric_parts = REWARD_SERIES_KEYS.get(key)
            if metric_parts is not None:
                metric_values[f"{'.'.join(metric_parts)}.mean"] = values

        hist_data = self._episode_hist_data(episode)
        if hist_data is None:
            return
        for tag, histogram_values in iter_lanistai_tensorboard_histogram_items(
            metric_values,
            profile=self._tensorboard_profile,
        ):
            hist_data.setdefault(tag, []).extend(histogram_values.tolist())
