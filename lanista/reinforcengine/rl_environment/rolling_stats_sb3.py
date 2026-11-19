# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Track rolling training statistics for Stable-Baselines3."""

from collections import deque

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
from lanista.reinforcengine.rl_metrics.aggregation import flatten_numeric_metrics
from lanista.reinforcengine.rl_metrics.channel_schema import OBSERVATION_STAT_ORDER


class StableBaselinesRollingStats(LoggerInterface):
    """Track runtime statistics for stable baselines rolling."""

    _OBSERVATION_ORDER = OBSERVATION_STAT_ORDER
    _METRIC_WINDOW_SCOPE = "terminal_step_per_completed_episode"

    _config_get = staticmethod(config_get)
    _coerce_number = staticmethod(coerce_number)
    _normalize_range = staticmethod(normalize_range)
    _threshold_lower_bounds = staticmethod(threshold_lower_bounds)
    _threshold_upper_bounds = staticmethod(threshold_upper_bounds)

    def __init__(self, training_conditions: NestedDictWrapper, **kwargs):

        """Initialize the stable baselines rolling stats and its owned state."""
        LoggerInterface.__init__(self, **kwargs)

        # Training conditions define rolling windows and switching thresholds.
        scenario_switch = self._config_get(training_conditions, "scenario_switch")
        self._lookback_window_size = self._config_get(
            scenario_switch,
            "lookback_window_size",
            self._config_get(training_conditions, "lookback_window_size"),
        )
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
        self._configure_curriculum_readiness(training_conditions)

        # Active-window repositories are kept separate from retained history.
        self._overall_values: list[dict] = []
        self._scenario_episode_baseline_index = 0
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

    def _configure_curriculum_readiness(
        self,
        training_conditions: NestedDictWrapper,
    ) -> None:
        """Load curriculum switching thresholds and its override window."""

        curriculum_switch = self._config_get(training_conditions, "curriculum_switch")
        self._curriculum_min_terminated_episodes = self._config_get(
            curriculum_switch,
            "min_terminated_episodes",
            0,
        )
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
        self._curriculum_entropy_range = self._config_get(curriculum_switch, "entropy_range", None)

    def add_episode(self, info, incoming_metrics, stats):
        # Link for understanding:
        # https://stable-baselines3.readthedocs.io/en/master/common/logger.html
        """Add the episode owned by this stable baselines rolling stats."""
        episode_info = info.get("episode", {})
        record = {
            # Episode data emitted by the environment.
            "episode_reward": info.get("ep_rew", 0),                                    # Total reward accumulated during the episode
            "episode_length_in_steps": episode_info.get("l", 0),                        # Number of steps in the episode (discrete time)
            "episode_length_in_miliseconds": episode_info.get("t", 0),                  # Duration of the episode in milliseconds (continuous time)
            "terminal_observation": info.get("terminal_observation"),                   # Final observation at episode end
            "terminated": bool(info.get("term", False)),
            "truncated": bool(info.get("trunc", False)),
            # Training metrics emitted by SB3.
            "learning_rate": incoming_metrics.get("train/learning_rate", 0),            # Learning rate used during training
            "entropy_loss": incoming_metrics.get("train/entropy_loss", 0),              # Entropy loss, encourages exploration
            "entropy": incoming_metrics.get("train/entropy", incoming_metrics.get("train/entropy_loss", 0)),
            "policy_gradient_loss": incoming_metrics.get("train/policy_gradient_loss", 0),
            "value_loss": incoming_metrics.get("train/value_loss", 0),
            "approx_kl": float(incoming_metrics.get("train/approx_kl", 0)),
            "clip_fraction": incoming_metrics.get("train/clip_fraction", 0),
            "loss": incoming_metrics.get("train/loss", 0),
            "explained_variance": incoming_metrics.get("train/explained_variance", 0),
            "std": incoming_metrics.get("train/std", 0),
            "n_updates": incoming_metrics.get("train/n_updates", 0),
            "clip_range": incoming_metrics.get("train/clip_range", 0),
            # Step counters maintained by the callback.
            "n_episode_steps": stats["n_episode_steps"],                                # Steps in current episode (discrete time)
            "n_rollout_steps": stats["n_rollout_steps"],                                # Steps in current rollout (sampling steps)
            "n_timesteps": stats["n_timesteps"],                                        # Total timesteps processed so far (aggregated discrete time)
            "total_timesteps": stats["total_timesteps"],                                # Cumulative timesteps across all episodes (global discrete time)
        }
        self._windowed_values.append(record)
        preferred_override_records(self).append(record)
        self._overall_values.append(record)
        self._logger.debug("New episode terminated, stats collected.")
        self._logger.info(f"Registered episode stats: {pretty_repr(record)}")

        lanistai_metrics = flatten_numeric_metrics(info.get("lanistai", {}))
        if lanistai_metrics:
            record["truncation_red_light_overdrive"] = bool(
                lanistai_metrics.get("episode_info.truncation_red_light_overdrive.sum", 0)
                or lanistai_metrics.get("episode_info.truncation_red_light_overdrive.mean", 0)
            )
            record["truncation_time_limit"] = bool(
                lanistai_metrics.get("episode_info.truncation_time_limit.sum", 0)
                or lanistai_metrics.get("episode_info.truncation_time_limit.mean", 0)
            )
            self._windowed_metric_snapshots.append(lanistai_metrics)
            self._historical_metric_snapshots.append(lanistai_metrics)

    log_window_metric_stats = _log_window_metric_stats

    scenario_ready = _scenario_ready

    metrics_ready = _metrics_ready

    def _scenario_rewards(self) -> tuple[float, ...]:
        """Return one reward for each completed SB3 episode in the window."""
        return tuple(record["episode_reward"] for record in self._windowed_values)

    def _overall_observed_episodes(self) -> int:
        """Count SB3 episodes observed since the active scenario began."""
        return len(self._scenario_episode_values())

    def reset(self):
        """Reset this stable baselines rolling stats to its initial operational state."""
        self._scenario_episode_baseline_index = len(self._overall_values)
        self._windowed_values.clear()
        preferred_override_records(self).clear()
        metric_window = getattr(self, "_windowed_metric_snapshots", None)
        if metric_window is not None:
            metric_window.clear()
        reset_scenario_readiness(self)

    def _scenario_episode_values(self) -> list[dict]:
        """Return retained SB3 records belonging to the active scenario."""

        values = getattr(self, "_overall_values", [])
        baseline = max(
            0,
            min(
                int(getattr(self, "_scenario_episode_baseline_index", 0)),
                len(values),
            ),
        )
        return values[baseline:]

    def _get_window_stats(self):
        """Return the window stats exposed by this stable baselines rolling stats."""
        return self._get_stats_for_records(self._windowed_values)

    def _get_preferred_override_window_stats(self, required_episodes: int):
        """Return up to the requested recent completed SB3 episodes."""

        records = list(preferred_override_records(self))
        return self._get_stats_for_records(records[-required_episodes:])

    @staticmethod
    def _get_stats_for_records(records):
        """Aggregate one explicit SB3 episode-record collection."""

        records = list(records)
        if not records:
            return None

        terminated_in_window = sum(1 for record in records if record.get("terminated"))
        truncated_in_window = sum(1 for record in records if record.get("truncated"))
        red_light_overdrive_in_window = sum(
            1 for record in records if record.get("truncation_red_light_overdrive")
        )
        time_limit_truncation_in_window = sum(
            1 for record in records if record.get("truncation_time_limit")
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

    def _overall_terminated_episodes(self) -> int:
        """Return active-scenario terminated SB3 episodes."""
        return sum(
            1
            for record in self._scenario_episode_values()
            if record.get("terminated")
        )

    def _overall_truncated_episodes(self) -> int:
        """Return active-scenario truncated SB3 episodes."""
        return sum(
            1
            for record in self._scenario_episode_values()
            if record.get("truncated")
        )

    def _all_terminated_episodes(self) -> int:
        """Return terminated SB3 episodes retained across scenario resets."""

        return sum(
            1
            for record in getattr(self, "_overall_values", [])
            if record.get("terminated")
        )

    def _all_truncated_episodes(self) -> int:
        """Return truncated SB3 episodes retained across scenario resets."""

        return sum(
            1
            for record in getattr(self, "_overall_values", [])
            if record.get("truncated")
        )

    episode_count_summary = _episode_count_summary
    curriculum_ready = _curriculum_ready
    training_progress_evidence = _training_progress_evidence
