# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Handle lifecycle callbacks for tensorboard."""

from typing import Protocol, cast

from stable_baselines3.common.callbacks import BaseCallback

from lanista.reinforcengine.rl_metrics.aggregation import (
    compute_stats,
    flatten_numeric_metrics,
    is_valid_number,
)
from lanista.reinforcengine.rl_metrics.tensorboard_schema import (
    TENSORBOARD_HISTOGRAM_CADENCE,
    TensorboardProfile,
    resolve_tensorboard_profile,
)
from lanista.reinforcengine.rl_metrics.tensorboard_surface import (
    SCENARIO_ACTIVE_TAG,
    iter_lanistai_tensorboard_histogram_items,
    should_log_lanistai_tensorboard_scalar,
    tensorboard_tag_from_flat_key,
)
from lanista.reinforcengine.rl_metrics.training_state_projection import (
    active_scenario_number,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_state_machine import (
    is_full_curriculum_state,
)


class _HistogramWriter(Protocol):
    """Define the typed boundary for histogram writer."""

    def add_histogram(
        self,
        tag: str,
        values: object,
        *,
        global_step: int,
    ) -> object:
        """Write one histogram sample."""

    def flush(self) -> object:
        """Flush buffered samples."""


class _SummaryWriterFactory(Protocol):
    """Define the typed boundary for summary writer factory."""

    def __call__(self, *, log_dir: str) -> _HistogramWriter:
        """Open one histogram writer."""


class _StateReader(Protocol):
    """Define the typed boundary for one shared training-state reader."""

    def get(self) -> object:
        """Return the latest curriculum state."""


SummaryWriter: _SummaryWriterFactory | None
try:
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional outside training.
    SummaryWriter = None
else:
    SummaryWriter = cast(_SummaryWriterFactory, _TorchSummaryWriter)


TENSORBOARD_STAT_NAMES = ("mean", "min", "max", "std", "p25", "p50", "p75", "range", "iqr")
EPISODE_TERMINATION_KEYS = frozenset(
    {
        "episode_info.episode_terminated.sum",
        "episode_info.episode_terminated.mean",
        "episode_info.episode_truncated.sum",
        "episode_info.episode_truncated.mean",
    }
)


def _tensorboard_histograms_enabled(value: bool | None = None) -> bool:
    """Return whether tensorboard histograms enabled applies to this tensorboard callback module."""
    if value is not None:
        return bool(value)
    return resolve_tensorboard_profile() is not TensorboardProfile.COMPACT


class StableBaselinesTensorboardCallback(BaseCallback):
    """Record LanistAI env info metrics through the SB3 logger."""

    def __init__(
        self,
        curriculum_index_state: object | None = None,
        scenario_index_state: object | None = None,
        record_rollout_stat_details: bool = False,
        record_rollout_histograms: bool | None = None,
        episode_info_cumulative_sums: dict[str, float] | None = None,
    ) -> None:
        """Initialize the stable baselines tensorboard callback and its owned state."""
        super().__init__()
        self._curriculum_index_state = curriculum_index_state
        self._scenario_index_state = scenario_index_state
        profile = resolve_tensorboard_profile()
        if (
            profile is TensorboardProfile.COMPACT
            and (record_rollout_stat_details or record_rollout_histograms is True)
        ):
            profile = TensorboardProfile.DIAGNOSTIC
        self._tensorboard_profile = profile
        self._record_rollout_stat_details = (
            profile is not TensorboardProfile.COMPACT
        )
        self._record_rollout_histograms_enabled = (
            profile is not TensorboardProfile.COMPACT
            and record_rollout_histograms is not False
        )
        self._histogram_window_count = 0
        self._histogram_writer: _HistogramWriter | None = None
        self._terminated_total = 0
        self._truncated_total = 0
        self._episode_info_cumulative_sums = (
            episode_info_cumulative_sums
            if episode_info_cumulative_sums is not None
            else {}
        )
        self._rollout_episode_info_values: dict[str, list[float]] = {}
        self._rollout_metric_values: dict[str, list[float]] = {}
        self._rollout_episode_intersections: list[float] = []

    @staticmethod
    def _get_nested(mapping: object, path: tuple[str, ...]) -> object | None:
        """Return the nested exposed by this stable baselines tensorboard callback."""
        value = mapping
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    def _curriculum_state(self) -> object | None:
        """Return the curriculum state value owned by this stable baselines tensorboard callback."""
        if self._curriculum_index_state is None:
            return None
        try:
            return cast(_StateReader, self._curriculum_index_state).get()
        except Exception:
            return None

    def _record_curriculum_state(self) -> None:
        """Record the curriculum state owned by this stable baselines tensorboard callback."""
        state = self._curriculum_state()
        if not isinstance(state, dict):
            self.logger.record("lanistai_curriculum/active_level", 0)
            self.logger.record("lanistai_curriculum/target_intersection", 0)
            self.logger.record("lanistai_curriculum/required_terminated_episodes", 0)
            self.logger.record("lanistai_curriculum/horizon_is_full", 0)
            return

        curriculum_index = state.get("curriculum_index")
        active_level = 0
        if is_valid_number(curriculum_index):
            active_level = int(cast(float | int | str, curriculum_index)) + 1

        target_value = state.get("target_value")
        target_intersection = float(target_value) if is_valid_number(target_value) else 0.0
        required_terminated = state.get("cumulative_min_terminated_episodes")
        required_terminated = float(required_terminated) if is_valid_number(required_terminated) else 0.0
        horizon_is_full = float(
            bool(
                state.get(
                    "horizon_is_full",
                    is_full_curriculum_state(state),
                )
            )
        )

        self.logger.record("lanistai_curriculum/active_level", active_level)
        self.logger.record("lanistai_curriculum/target_intersection", target_intersection)
        self.logger.record("lanistai_curriculum/required_terminated_episodes", required_terminated)
        self.logger.record("lanistai_curriculum/horizon_is_full", horizon_is_full)

    def _record_scenario_state(self) -> None:
        """Record the shared one-based active scenario number."""

        state = getattr(self, "_scenario_index_state", None)
        try:
            scenario_index = cast(_StateReader, state).get()
        except Exception:
            scenario_index = None
        self.logger.record(
            SCENARIO_ACTIVE_TAG,
            active_scenario_number(scenario_index),
        )

    def _get_histogram_writer(self) -> _HistogramWriter | None:
        """Return the histogram writer exposed by this stable baselines tensorboard callback."""
        if not getattr(self, "_record_rollout_histograms_enabled", False):
            return None
        if SummaryWriter is None:
            return None
        if self._histogram_writer is not None:
            return self._histogram_writer

        sb3_logger = getattr(getattr(self, "model", None), "logger", None)
        log_dir = getattr(sb3_logger, "dir", None)
        if not log_dir:
            return None
        self._histogram_writer = SummaryWriter(log_dir=log_dir)
        return self._histogram_writer

    def _record_rollout_histograms(self) -> None:
        """Record the rollout histograms owned by this stable baselines tensorboard callback."""
        self._histogram_window_count += 1
        cadence = (
            1
            if self._tensorboard_profile is TensorboardProfile.FULL
            else TENSORBOARD_HISTOGRAM_CADENCE
        )
        if self._histogram_window_count % cadence:
            return
        writer = self._get_histogram_writer()
        if writer is None:
            return
        for tag, values in iter_lanistai_tensorboard_histogram_items(
            self._rollout_metric_values,
            profile=self._tensorboard_profile,
        ):
            writer.add_histogram(tag, values, global_step=self.num_timesteps)
        writer.flush()

    def _ensure_episode_accumulators(self) -> None:
        """Ensure the episode accumulators owned by this stable baselines tensorboard callback."""
        if not hasattr(self, "_rollout_episode_info_values"):
            self._rollout_episode_info_values = {}
        if not hasattr(self, "_episode_info_cumulative_sums"):
            self._episode_info_cumulative_sums = {}

    def _record_completed_episode(
        self,
        lanistai_info: object,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record the completed episode owned by this stable baselines tensorboard callback."""
        for key, value in {
            "episode_info.episode_terminated.sum": float(terminated),
            "episode_info.episode_terminated.mean": float(terminated),
            "episode_info.episode_truncated.sum": float(truncated),
            "episode_info.episode_truncated.mean": float(truncated),
        }.items():
            self._rollout_episode_info_values.setdefault(key, []).append(value)

        reached_intersection = self._get_nested(
            lanistai_info,
            ("vehicle_info", "reached_intersection_number", "mean"),
        )
        if is_valid_number(reached_intersection):
            self._rollout_episode_intersections.append(float(reached_intersection))

    def _record_flattened_step_metrics(
        self,
        lanistai_info: object,
        *,
        episode_done: bool,
    ) -> None:
        """Record the flattened step metrics owned by this stable baselines tensorboard callback."""
        for key, value in flatten_numeric_metrics(lanistai_info).items():
            if key.startswith("episode_info."):
                if key in EPISODE_TERMINATION_KEYS:
                    continue
                if (
                    episode_done
                    and is_valid_number(value)
                    and should_log_lanistai_tensorboard_scalar(
                        key,
                        profile=self._tensorboard_profile,
                    )
                ):
                    self._rollout_episode_info_values.setdefault(key, []).append(float(value))
                continue
            if (
                is_valid_number(value)
                and should_log_lanistai_tensorboard_scalar(
                    key,
                    profile=self._tensorboard_profile,
                )
            ):
                self._rollout_metric_values.setdefault(key, []).append(float(value))

    def _record_step_info(self, info: object) -> tuple[int, int]:
        """Record the step info owned by this stable baselines tensorboard callback."""
        if not isinstance(info, dict):
            return 0, 0
        terminated = bool(info.get("term", False))
        truncated = bool(info.get("trunc", False))
        lanistai_info = info.get("lanistai", {})
        episode_done = terminated or truncated
        if episode_done:
            self._record_completed_episode(
                lanistai_info,
                terminated=terminated,
                truncated=truncated,
            )
        self._record_flattened_step_metrics(
            lanistai_info,
            episode_done=episode_done,
        )
        return int(terminated), int(truncated)

    def _on_step(self) -> bool:
        """Handle the step event for this stable baselines tensorboard callback."""
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        self._ensure_episode_accumulators()

        terminated_this_step = 0
        truncated_this_step = 0
        for info in infos:
            terminated, truncated = self._record_step_info(info)
            terminated_this_step += terminated
            truncated_this_step += truncated

        self._terminated_total += terminated_this_step
        self._truncated_total += truncated_this_step
        return True

    def _record_episode_info_scalar(self, flat_key: str, values: list[float]) -> None:
        """Record the episode info scalar owned by this stable baselines tensorboard callback."""
        if not should_log_lanistai_tensorboard_scalar(
            flat_key,
            profile=self._tensorboard_profile,
        ):
            return
        if flat_key.endswith(".sum"):
            value = self._episode_info_cumulative_sums.get(flat_key, 0.0) + sum(values)
            self._episode_info_cumulative_sums[flat_key] = value
        else:
            value = sum(values) / len(values)
        self.logger.record(tensorboard_tag_from_flat_key(flat_key), value)

    def _record_episode_info_scalars(self) -> None:
        """Record the episode info scalars owned by this stable baselines tensorboard callback."""
        for flat_key, values in getattr(self, "_rollout_episode_info_values", {}).items():
            if values:
                self._record_episode_info_scalar(flat_key, values)

    def _record_metric_statistics(self, key: str, values: list[float]) -> None:
        """Record the metric statistics owned by this stable baselines tensorboard callback."""
        stats = compute_stats(values)
        if stats is None:
            return

        stats = {
            **stats,
            "range": stats["max"] - stats["min"],
            "iqr": stats["p75"] - stats["p25"],
        }
        base_key = key[:-5] if key.endswith(".mean") else key
        for stat_name in TENSORBOARD_STAT_NAMES:
            flat_stat_key = f"{base_key}.{stat_name}"
            if not should_log_lanistai_tensorboard_scalar(
                flat_stat_key,
                include_stat_details=getattr(self, "_record_rollout_stat_details", False),
                profile=self._tensorboard_profile,
            ):
                continue
            stat_key = tensorboard_tag_from_flat_key(flat_stat_key)
            self.logger.record(stat_key, stats[stat_name])

    def _record_rollout_metric_statistics(self) -> None:
        """Record the rollout metric statistics owned by this stable baselines tensorboard
        callback.
        """
        for key, values in self._rollout_metric_values.items():
            if not values:
                continue
            if key.startswith("episode_setup.") and key.endswith(".value"):
                if should_log_lanistai_tensorboard_scalar(
                    key,
                    profile=self._tensorboard_profile,
                ):
                    self.logger.record(
                        tensorboard_tag_from_flat_key(key),
                        sum(values) / len(values),
                    )
                continue
            self._record_metric_statistics(key, values)

    def _record_episode_intersection_summary(self) -> None:
        """Record the episode intersection summary owned by this stable baselines tensorboard
        callback.
        """
        if self._rollout_episode_intersections:
            self.logger.record("lanistai_vehicle/reached_intersection_number/episode_max", max(self._rollout_episode_intersections))

    def _clear_rollout_accumulators(self) -> None:
        """Clear the rollout accumulators owned by this stable baselines tensorboard callback."""
        self._rollout_metric_values.clear()
        if hasattr(self, "_rollout_episode_info_values"):
            self._rollout_episode_info_values.clear()
        self._rollout_episode_intersections.clear()

    def _on_rollout_end(self) -> None:
        """Handle the rollout end event for this stable baselines tensorboard callback."""
        self._record_curriculum_state()
        self._record_scenario_state()
        self._record_episode_info_scalars()
        self._record_rollout_metric_statistics()
        self._record_rollout_histograms()
        self._record_episode_intersection_summary()
        self._clear_rollout_accumulators()
