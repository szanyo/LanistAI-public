# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Track rolling training statistics for RLlib."""

import os
from collections import deque
from dataclasses import dataclass
from pprint import pformat

from rich.pretty import pretty_repr

from lanista.foundation.configuration import NestedDictWrapper
from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.rl_environment.rolling_stats_adapter_projection import (
    configure_scenario_readiness,
    preferred_override_records,
    reset_scenario_readiness,
)
from lanista.reinforcengine.rl_environment.rolling_stats_adapter_projection import (
    curriculum_ready as _curriculum_ready,
)
from lanista.reinforcengine.rl_environment.rolling_stats_adapter_projection import (
    episode_count_summary as _episode_count_summary,
)
from lanista.reinforcengine.rl_environment.rolling_stats_adapter_projection import (
    metrics_ready as _metrics_ready,
)
from lanista.reinforcengine.rl_environment.rolling_stats_adapter_projection import (
    scenario_ready as _scenario_ready,
)
from lanista.reinforcengine.rl_environment.rolling_stats_adapter_projection import (
    training_progress_evidence as _training_progress_evidence,
)
from lanista.reinforcengine.rl_environment.rolling_stats_configuration import (
    coerce_number,
    config_get,
    normalize_range,
    scenario_minimum_episode_floor,
    threshold_lower_bounds,
    threshold_upper_bounds,
)
from lanista.reinforcengine.rl_environment.rolling_stats_reporting import (
    log_window_metric_stats as _log_window_metric_stats,
)
from lanista.reinforcengine.rl_metrics.channel_schema import OBSERVATION_STAT_ORDER
from lanista.reinforcengine.rl_metrics.tensorboard_surface import (
    canonical_training_metrics_from_result,
)


@dataclass(frozen=True, slots=True)
class _EpisodeCounters:
    """One lifetime or per-window Ray episode counter snapshot."""

    terminated: int
    truncated: int
    red_light_overdrive: int
    time_limit_truncation: int

    @property
    def episode_count(self) -> int:
        """Return the episode count exposed by this episode counters."""
        return self.terminated + self.truncated


@dataclass(frozen=True, slots=True)
class _EpisodeIngestion:
    """Validated Ray result payload and its normalized counter projections."""

    env_runners: dict
    totals: _EpisodeCounters
    deltas: _EpisodeCounters


class RayRLlibRollingStats(LoggerInterface):
    """Track runtime statistics for ray r llib rolling."""

    _OBSERVATION_ORDER = OBSERVATION_STAT_ORDER
    _METRIC_WINDOW_SCOPE = "rllib_reduced_train_result"
    _FULL_TRAIN_RESULT_LOG_ENV = "LANISTAI_RAY_LOG_FULL_TRAIN_RESULT"

    _config_get = staticmethod(config_get)
    _coerce_number = staticmethod(coerce_number)
    _normalize_range = staticmethod(normalize_range)
    _threshold_lower_bounds = staticmethod(threshold_lower_bounds)
    _threshold_upper_bounds = staticmethod(threshold_upper_bounds)

    def __init__(self, training_conditions: NestedDictWrapper, **kwargs):
        """Initialize the ray r llib rolling stats and its owned state."""
        LoggerInterface.__init__(self, **kwargs)

        # Training conditions define rolling windows and switching thresholds.
        scenario_switch = training_conditions.scenario_switch
        self._lookback_window_size = scenario_switch.lookback_window_size
        self._min_episodes_for_switch_per_scenario = (
            scenario_minimum_episode_floor(training_conditions, scenario_switch)
        )
        self._min_terminated_episodes_for_switch_per_scenario = self._config_get(
            scenario_switch,
            "min_terminated_episodes_for_switch_per_scenario",
            0,
        )
        configure_scenario_readiness(self, scenario_switch)

        self._entropy_loss_threshold = training_conditions.metrics.entropy_loss_threshold
        self._policy_std_threshold = training_conditions.metrics.policy_std_threshold
        self._approx_kl_threshold = training_conditions.metrics.approx_kl_threshold
        self._clip_fraction_threshold = training_conditions.metrics.clip_fraction_threshold
        self._explained_variance_threshold = training_conditions.metrics.explained_variance_threshold
        self._episode_reward_stability = training_conditions.metrics.episode_reward_stability

        curriculum_switch = training_conditions.curriculum_switch
        self._curriculum_min_terminated_episodes = curriculum_switch.min_terminated_episodes
        self._curriculum_termination_ratio_threshold = self._config_get(
            curriculum_switch,
            "termination_ratio_threshold",
            self._config_get(curriculum_switch, "min_termination_rate", 0.0),
        )
        self._curriculum_red_light_overdrive_rate_threshold = self._config_get(
            curriculum_switch,
            "red_light_overdrive_rate_threshold",
            None,
        )
        self._curriculum_time_limit_truncation_rate_threshold = self._config_get(
            curriculum_switch,
            "time_limit_truncation_rate_threshold",
            None,
        )
        self._curriculum_preferred_override_window_size = self._coerce_number(
            self._config_get(
                curriculum_switch,
                "preferred_override_window_size",
                0,
            ),
            0,
            int,
        )
        self._curriculum_entropy_range = curriculum_switch.entropy_range

        # Active-window repositories are kept separate from retained history.
        self._overall_values: list[dict] = []
        self._windowed_values: deque[dict] = deque(maxlen=self._lookback_window_size)
        preferred_override_capacity = max(
            self._coerce_number(
                self._scenario_preferred_override_window_size,
                0,
                int,
            ),
            self._curriculum_preferred_override_window_size,
        )
        self._preferred_override_windowed_values: deque[dict] = deque(
            maxlen=preferred_override_capacity,
        )
        self._windowed_metric_snapshots: deque[dict] = deque(maxlen=self._lookback_window_size)
        self._historical_metric_snapshots: list[dict] = []

        # These counters distinguish episode termination and truncation causes.
        self._recorded_episode_termination_counter = 0
        self._recorded_episode_truncation_counter = 0
        self._recorded_red_light_overdrive_counter = 0
        self._recorded_time_limit_truncation_counter = 0
        self._scenario_episode_counter_baseline = _EpisodeCounters(0, 0, 0, 0)

    @staticmethod
    def _get_env_runners(train_result_info: dict) -> dict | None:
        """Return the env runners exposed by this ray r llib rolling stats."""
        if "env_runners" in train_result_info:
            return train_result_info.get("env_runners")
        if "env_runner_results" in train_result_info:
            return train_result_info.get("env_runner_results")
        if "sampler_results" in train_result_info:
            return train_result_info.get("sampler_results")
        return None

    @staticmethod
    def _get_learners(train_result_info: dict) -> dict:
        """Return the learners exposed by this ray r llib rolling stats."""
        if "learners" in train_result_info:
            return train_result_info.get("learners", {})
        info = train_result_info.get("info", {})
        return info.get("learner", {})

    def _extract_episode_totals(
        self,
        train_result_info: dict,
    ) -> tuple[dict, _EpisodeCounters] | None:
        """Return the extract episode totals value owned by this ray r llib rolling stats."""
        env_runners = self._get_env_runners(train_result_info)
        if not env_runners:
            self._logger.warning(
                "Missing env_runners in train result; "
                "skipping rolling stats update."
            )
            return None

        has_metric_namespace = isinstance(env_runners.get("lanistai"), dict) or any(
            str(key).startswith("lanistai_") for key in env_runners
        )
        if not has_metric_namespace:
            self._logger.warning(
                "Missing lanistai metrics in env_runners; "
                "skipping rolling stats update."
            )
            return None

        metrics = canonical_training_metrics_from_result(env_runners)
        sum_episode_terminated = metrics.get(
            "episode_info.episode_terminated.sum"
        )
        sum_episode_truncated = metrics.get(
            "episode_info.episode_truncated.sum"
        )
        if sum_episode_terminated is None or sum_episode_truncated is None:
            self._logger.warning(
                "Missing episode termination counters; "
                "skipping rolling stats update."
            )
            return None

        return env_runners, _EpisodeCounters(
            terminated=self._coerce_number(
                sum_episode_terminated,
                0,
                int,
            ),
            truncated=self._coerce_number(
                sum_episode_truncated,
                0,
                int,
            ),
            red_light_overdrive=self._coerce_number(
                metrics.get(
                    "episode_info.truncation_red_light_overdrive.sum"
                ),
                0,
                int,
            ),
            time_limit_truncation=self._coerce_number(
                metrics.get("episode_info.truncation_time_limit.sum"),
                0,
                int,
            ),
        )

    def _previous_episode_counters(self) -> _EpisodeCounters:
        """Return the previous episode counters value owned by this ray r llib rolling stats."""
        return _EpisodeCounters(
            terminated=self._coerce_number(
                getattr(self, "_recorded_episode_termination_counter", 0),
                0,
                int,
            ),
            truncated=self._coerce_number(
                getattr(self, "_recorded_episode_truncation_counter", 0),
                0,
                int,
            ),
            red_light_overdrive=self._coerce_number(
                getattr(self, "_recorded_red_light_overdrive_counter", 0),
                0,
                int,
            ),
            time_limit_truncation=self._coerce_number(
                getattr(self, "_recorded_time_limit_truncation_counter", 0),
                0,
                int,
            ),
        )

    @staticmethod
    def _episode_deltas(
        totals: _EpisodeCounters,
        previous: _EpisodeCounters,
    ) -> _EpisodeCounters:
        """Return the episode deltas value owned by this ray r llib rolling stats."""
        if (
            totals.terminated < previous.terminated
            or totals.truncated < previous.truncated
        ):
            previous = _EpisodeCounters(0, 0, 0, 0)
        return _EpisodeCounters(
            terminated=max(0, totals.terminated - previous.terminated),
            truncated=max(0, totals.truncated - previous.truncated),
            red_light_overdrive=max(
                0,
                totals.red_light_overdrive
                - previous.red_light_overdrive,
            ),
            time_limit_truncation=max(
                0,
                totals.time_limit_truncation
                - previous.time_limit_truncation,
            ),
        )

    def _publish_episode_counters(
        self,
        totals: _EpisodeCounters,
    ) -> None:
        """Publish the episode counters owned by this ray r llib rolling stats."""
        self._recorded_episode_termination_counter = totals.terminated
        self._recorded_episode_truncation_counter = totals.truncated
        self._recorded_red_light_overdrive_counter = (
            totals.red_light_overdrive
        )
        self._recorded_time_limit_truncation_counter = (
            totals.time_limit_truncation
        )

    def _ingest_episode_counters(
        self,
        train_result_info: dict,
    ) -> _EpisodeIngestion | None:
        """Return the ingest episode counters value owned by this ray r llib rolling stats."""
        extracted = self._extract_episode_totals(train_result_info)
        if extracted is None:
            return None
        env_runners, totals = extracted
        deltas = self._episode_deltas(
            totals,
            self._previous_episode_counters(),
        )
        self._publish_episode_counters(totals)
        return _EpisodeIngestion(
            env_runners=env_runners,
            totals=totals,
            deltas=deltas,
        )

    def _build_episode_record(
        self,
        train_result_info: dict,
        ingestion: _EpisodeIngestion,
    ) -> dict[str, int | float]:
        """Build the episode record required by this ray r llib rolling stats."""
        learners = self._get_learners(train_result_info)
        ego_learner = learners.get("EGO_policy", {})
        episode_return_mean = ingestion.env_runners.get(
            "episode_return_mean"
        )
        episode_duration_sec_mean = ingestion.env_runners.get(
            "episode_duration_sec_mean"
        )
        return {
            "terminated_episodes_total": ingestion.totals.terminated,
            "truncated_episodes_total": ingestion.totals.truncated,
            "terminated_episodes_delta": ingestion.deltas.terminated,
            "truncated_episodes_delta": ingestion.deltas.truncated,
            "red_light_overdrive_total": (
                ingestion.totals.red_light_overdrive
            ),
            "time_limit_truncation_total": (
                ingestion.totals.time_limit_truncation
            ),
            "red_light_overdrive_delta": (
                ingestion.deltas.red_light_overdrive
            ),
            "time_limit_truncation_delta": (
                ingestion.deltas.time_limit_truncation
            ),
            "episode_reward_mean": (
                episode_return_mean
                if episode_return_mean is not None
                else float("nan")
            ),
            "episode_length_in_miliseconds": (episode_duration_sec_mean * 1000)
            if episode_duration_sec_mean is not None
            else float("nan"),
            "entropy": ego_learner.get("entropy", float("nan")),
            "vf_explained_var": ego_learner.get(
                "vf_explained_var",
                float("nan"),
            ),
            "vf_loss": ego_learner.get("vf_loss", float("nan")),
            "policy_loss": ego_learner.get("policy_loss", float("nan")),
            "mean_kl_loss": ego_learner.get("mean_kl_loss", float("nan")),
            "clip_fraction": ego_learner.get("clip_fraction", float("nan")),
            "total_loss": ego_learner.get("total_loss", float("nan")),
        }

    def _log_episode_record(
        self,
        train_result_info: dict,
        record: dict[str, int | float],
        deltas: _EpisodeCounters,
    ) -> None:
        """Log the episode record owned by this ray r llib rolling stats."""
        if os.environ.get(self._FULL_TRAIN_RESULT_LOG_ENV) == "1":
            self._logger.debug(pformat(train_result_info))
        self._logger.debug(
            f"{deltas.terminated} episode terminated "
            "in current Ray train result"
        )
        self._logger.debug(
            f"{deltas.truncated} episode truncated "
            "in current Ray train result"
        )
        self._logger.info(
            "Stats successfully collected for previous training epoch."
        )
        self._logger.info(f"Registered episode stats: {pretty_repr(record)}")

    def _capture_lanistai_metric_snapshot(
        self,
        train_result_info: dict,
    ) -> None:
        """Capture the lanistai metric snapshot required by this ray r llib rolling stats."""
        lanistai_metrics = self._extract_lanistai_metrics(train_result_info)
        if not lanistai_metrics:
            return
        self._windowed_metric_snapshots.append(lanistai_metrics)
        self._historical_metric_snapshots.append(lanistai_metrics)

    def add_episode(self, train_result_info: dict) -> bool:
        """Add the episode owned by this ray r llib rolling stats."""
        ingestion = self._ingest_episode_counters(train_result_info)
        if ingestion is None or ingestion.deltas.episode_count <= 0:
            return False

        record = self._build_episode_record(
            train_result_info,
            ingestion,
        )
        self._windowed_values.append(record)
        preferred_override_records(self).append(record)
        self._overall_values.append(record)
        self._log_episode_record(
            train_result_info,
            record,
            ingestion.deltas,
        )
        self._capture_lanistai_metric_snapshot(train_result_info)
        return True

    def _extract_lanistai_metrics(self, train_result_info: dict) -> dict:
        """Return the extract lanistai metrics value owned by this ray r llib rolling stats."""
        env_runners = self._get_env_runners(train_result_info) or {}
        return canonical_training_metrics_from_result(env_runners)

    log_window_metric_stats = _log_window_metric_stats

    scenario_ready = _scenario_ready

    metrics_ready = _metrics_ready

    def _scenario_rewards(self) -> tuple[float, ...]:
        """Return Ray result means without treating iterations as episodes."""
        return tuple(record["episode_reward_mean"] for record in self._windowed_values)

    def _overall_observed_episodes(self) -> int:
        """Count distributed episodes completed in the active scenario."""
        return self._scenario_episode_counters().episode_count

    def reset(self):
        """Reset this ray r llib rolling stats to its initial operational state."""
        self._scenario_episode_counter_baseline = self._latest_episode_counters()
        self._windowed_values.clear()
        preferred_override_records(self).clear()
        metric_window = getattr(self, "_windowed_metric_snapshots", None)
        if metric_window is not None:
            metric_window.clear()
        reset_scenario_readiness(self)

    def _latest_episode_counters(self) -> _EpisodeCounters:
        """Return the latest cumulative episode counters published by RLlib."""

        overall_values = getattr(self, "_overall_values", [])
        if not overall_values:
            return _EpisodeCounters(0, 0, 0, 0)
        latest = overall_values[-1]
        return _EpisodeCounters(
            terminated=self._coerce_number(
                latest.get("terminated_episodes_total"), 0, int
            ),
            truncated=self._coerce_number(
                latest.get("truncated_episodes_total"), 0, int
            ),
            red_light_overdrive=self._coerce_number(
                latest.get("red_light_overdrive_total"), 0, int
            ),
            time_limit_truncation=self._coerce_number(
                latest.get("time_limit_truncation_total"), 0, int
            ),
        )

    def _scenario_episode_counters(self) -> _EpisodeCounters:
        """Project cumulative RLlib counters onto the active scenario."""

        totals = self._latest_episode_counters()
        baseline = getattr(
            self,
            "_scenario_episode_counter_baseline",
            _EpisodeCounters(0, 0, 0, 0),
        )
        if (
            totals.terminated < baseline.terminated
            or totals.truncated < baseline.truncated
        ):
            baseline = _EpisodeCounters(0, 0, 0, 0)
            self._scenario_episode_counter_baseline = baseline
        return _EpisodeCounters(
            terminated=max(0, totals.terminated - baseline.terminated),
            truncated=max(0, totals.truncated - baseline.truncated),
            red_light_overdrive=max(
                0,
                totals.red_light_overdrive - baseline.red_light_overdrive,
            ),
            time_limit_truncation=max(
                0,
                totals.time_limit_truncation
                - baseline.time_limit_truncation,
            ),
        )

    def _get_window_stats(self):
        """Return the window stats exposed by this ray r llib rolling stats."""
        if len(self._windowed_values) == 0:
            return None

        if any("terminated_episodes_delta" in record for record in self._windowed_values):
            terminated_in_window = sum(
                self._coerce_number(record.get("terminated_episodes_delta"), 0, int)
                for record in self._windowed_values
            )
            truncated_in_window = sum(
                self._coerce_number(record.get("truncated_episodes_delta"), 0, int)
                for record in self._windowed_values
            )
            red_light_overdrive_in_window = sum(
                self._coerce_number(record.get("red_light_overdrive_delta"), 0, int)
                for record in self._windowed_values
            )
            time_limit_truncation_in_window = sum(
                self._coerce_number(record.get("time_limit_truncation_delta"), 0, int)
                for record in self._windowed_values
            )
        else:
            last = self._windowed_values[-1]
            last_idx = len(self._overall_values) - 1
            first_idx = last_idx - (len(self._windowed_values) - 1)
            prev_idx = first_idx - 1

            if prev_idx >= 0:
                baseline = self._overall_values[prev_idx]
                terminated_base = self._coerce_number(baseline.get("terminated_episodes_total"), 0, int)
                truncated_base = self._coerce_number(baseline.get("truncated_episodes_total"), 0, int)
                red_light_overdrive_base = self._coerce_number(baseline.get("red_light_overdrive_total"), 0, int)
                time_limit_truncation_base = self._coerce_number(baseline.get("time_limit_truncation_total"), 0, int)
            else:
                terminated_base = 0
                truncated_base = 0
                red_light_overdrive_base = 0
                time_limit_truncation_base = 0

            terminated_in_window = max(
                0,
                self._coerce_number(last.get("terminated_episodes_total"), 0, int) - terminated_base,
            )
            truncated_in_window = max(
                0,
                self._coerce_number(last.get("truncated_episodes_total"), 0, int) - truncated_base,
            )
            red_light_overdrive_in_window = max(
                0,
                self._coerce_number(last.get("red_light_overdrive_total"), 0, int) - red_light_overdrive_base,
            )
            time_limit_truncation_in_window = max(
                0,
                self._coerce_number(last.get("time_limit_truncation_total"), 0, int) - time_limit_truncation_base,
            )
        total_in_window = terminated_in_window + truncated_in_window

        termination_rate = (terminated_in_window / total_in_window) if total_in_window > 0 else 0.0
        red_light_overdrive_rate = (
            red_light_overdrive_in_window / total_in_window
            if total_in_window > 0 else 0.0
        )
        time_limit_truncation_rate = (
            time_limit_truncation_in_window / total_in_window
            if total_in_window > 0 else 0.0
        )

        return {
            "terminated_in_window": terminated_in_window,
            "truncated_in_window": truncated_in_window,
            "termination_rate": termination_rate,
            "red_light_overdrive_in_window": red_light_overdrive_in_window,
            "red_light_overdrive_rate": red_light_overdrive_rate,
            "time_limit_truncation_in_window": time_limit_truncation_in_window,
            "time_limit_truncation_rate": time_limit_truncation_rate,
        }

    def _get_preferred_override_window_stats(self, required_episodes: int):
        """Aggregate recent Ray result deltas until enough episodes are covered."""

        selected_records = []
        observed_episodes = 0
        for record in reversed(preferred_override_records(self)):
            selected_records.append(record)
            observed_episodes += self._coerce_number(
                record.get("terminated_episodes_delta"),
                0,
                int,
            ) + self._coerce_number(
                record.get("truncated_episodes_delta"),
                0,
                int,
            )
            if observed_episodes >= required_episodes:
                break
        if not selected_records:
            return None

        terminated = sum(
            self._coerce_number(record.get("terminated_episodes_delta"), 0, int)
            for record in selected_records
        )
        truncated = sum(
            self._coerce_number(record.get("truncated_episodes_delta"), 0, int)
            for record in selected_records
        )
        red_light = sum(
            self._coerce_number(record.get("red_light_overdrive_delta"), 0, int)
            for record in selected_records
        )
        time_limit = sum(
            self._coerce_number(record.get("time_limit_truncation_delta"), 0, int)
            for record in selected_records
        )
        total = terminated + truncated
        return {
            "terminated_in_window": terminated,
            "truncated_in_window": truncated,
            "termination_rate": terminated / total if total else 0.0,
            "red_light_overdrive_in_window": red_light,
            "red_light_overdrive_rate": red_light / total if total else 0.0,
            "time_limit_truncation_in_window": time_limit,
            "time_limit_truncation_rate": time_limit / total if total else 0.0,
        }

    def _overall_terminated_episodes(self) -> int:
        """Return active-scenario terminated distributed episodes."""
        return self._scenario_episode_counters().terminated

    def _overall_truncated_episodes(self) -> int:
        """Return active-scenario truncated distributed episodes."""
        return self._scenario_episode_counters().truncated

    def _all_terminated_episodes(self) -> int:
        """Return terminated Ray episodes retained across scenario resets."""

        return self._latest_episode_counters().terminated

    def _all_truncated_episodes(self) -> int:
        """Return truncated Ray episodes retained across scenario resets."""

        return self._latest_episode_counters().truncated

    episode_count_summary = _episode_count_summary
    curriculum_ready = _curriculum_ready
    training_progress_evidence = _training_progress_evidence
