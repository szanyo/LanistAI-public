# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate the lifecycle of checkpoint."""

import csv
import os
import uuid
from collections.abc import Callable, Mapping

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from lanista.reinforcengine.rl_metrics.aggregation import (
    extract_current_observation_from_dict,
    filter_episode_info_stats,
    filter_episode_setup_stats,
    filter_observation_stats,
    filter_reward_stats,
    flatten_numeric_metrics,
    historical_stat_groups_from_snapshots,
)
from lanista.reinforcengine.rl_metrics.channel_schema import (
    POLICY_OBSERVATION_PARAM_LENGTHS,
    POLICY_OBSERVATION_PARAM_NAMES,
)
from lanista.reinforcengine.rl_metrics.checkpoint_observation_csv import (
    CHECKPOINT_OBSERVATION_CSV_COLUMNS,
    write_checkpoint_observation_csv,
)
from lanista.reinforcengine.rl_model.checkpoints.metadata import CheckpointStateReadError
from lanista.reinforcengine.rl_model.checkpoints.runtime_repository import (
    CheckpointArtifactKind,
    RuntimeCheckpointMutation,
    RuntimeCheckpointRepository,
)
from lanista.reinforcengine.rl_technologies.sb3.checkpoint_resolution import (
    reserve_sb3_checkpoint_path as _reserve_sb3_checkpoint_path,
)
from lanista.reinforcengine.rl_technologies.sb3.model_policy import (
    SB3CheckpointObservationRow,
)

SB3_OBSERVATION_CSV_COLUMNS = CHECKPOINT_OBSERVATION_CSV_COLUMNS

CHECKPOINT_METRIC_STATS_COLUMNS = [
    "metric",
    "mean",
    "min",
    "max",
    "std",
    "p25",
    "p50",
    "p75",
    "range",
    "iqr",
]

QualityReporter = Callable[[str, BaseException], None]
_OPTIONAL_TELEMETRY_ERRORS = (
    OSError,
    csv.Error,
    RuntimeError,
    TypeError,
    ValueError,
)


def _warn_quality_degradation(
    logger,
    event: str,
    exc: BaseException,
    *,
    quality: str = "degraded",
) -> None:
    """Report the quality degradation owned by this checkpoint lifecycle
module."""
    if logger is None:
        return
    logger.warning(
        f"event={event} quality={quality} "
        f"error_type={type(exc).__name__} error={exc}"
    )


def effective_checkpoint_step(raw_step: int, base_step_offset: int = 0) -> int:
    """Return the effective checkpoint step value owned by this checkpoint lifecycle module."""
    raw_step = int(raw_step or 0)
    base_step_offset = int(base_step_offset or 0)
    if base_step_offset > 0 and raw_step <= base_step_offset:
        return base_step_offset + raw_step
    return raw_step


def build_sb3_checkpoint_name(scenario_num: int, step_num: int, *, manual: bool = False) -> str:
    """Build the sb3 checkpoint name required by this checkpoint lifecycle module."""
    base_name = f"scenario_{int(scenario_num)}_checkpoint_{int(step_num)}_steps"
    suffix = "_manual" if manual else ""
    return f"{base_name}{suffix}.zip"


def update_last_checkpoint_metadata(
    mutation: RuntimeCheckpointMutation,
    model_package_directory: str,
    checkpoint_name: str,
    checkpoint_kind: str,
    logger=None,
) -> None:
    """Update the last checkpoint metadata owned by this checkpoint lifecycle module."""
    _ = model_package_directory
    try:
        mutation.update_last_checkpoint_metadata(
            checkpoint_name,
            checkpoint_kind,
            logger=logger,
            raise_on_error=True,
        )
    except Exception as exc:
        _warn_quality_degradation(
            logger,
            "checkpoint_session_metadata_update_failed",
            exc,
            quality="invalid",
        )
        raise


def _reward_rows(rewards) -> list[float]:
    """Return the reward rows value owned by this checkpoint lifecycle module."""
    if rewards is None:
        return []
    arr = np.asarray(rewards, dtype=float)
    if arr.size == 0:
        return []
    return [float(value) for value in arr.reshape(-1)]


def _reward_component_fields(info: dict | None) -> dict:
    """Return the reward component fields value owned by this checkpoint lifecycle module."""
    if not isinstance(info, dict):
        return {}

    flattened = flatten_numeric_metrics(info.get("lanistai", {}).get("rewards", {}))
    fields = {}
    for key, value in flattened.items():
        csv_key = f"reward_{key.replace('.', '_')}"
        fields[csv_key] = float(value)
    return fields


def _observation_diagnostic_fields(info: dict | None) -> dict:
    """Return the observation diagnostic fields value owned by this checkpoint lifecycle module."""
    if not isinstance(info, dict):
        return {}

    flattened = flatten_numeric_metrics(info.get("lanistai", {}).get("observations", {}))
    fields = {}
    for column in (
            "tl_phase_remaining_delta_norm",
            "tl_phase_remaining_raw_delta_norm",
            "tl_timing_context_changed",
            "tl_phase_remaining_monotonic_violation",
    ):
        value = flattened.get(f"{column}.stats.mean")
        if value is not None and np.isfinite(float(value)):
            fields[column] = float(value)
    return fields


def _diagnostic_trace_fields(info: dict | None) -> dict[str, object]:
    """Return one explicitly gated model-owned diagnostic trace row."""

    if not isinstance(info, dict):
        return {}
    trace = info.get("lanistai_diagnostic_trace")
    if not isinstance(trace, Mapping):
        return {}
    fields = {str(key): value for key, value in trace.items()}
    if fields:
        fields["_diagnostic_trace_fields"] = tuple(fields)
    return fields


def _mapping_observation_values(
    observation: object,
    quality_reporter: QualityReporter | None,
) -> tuple[bool, dict[str, float]]:
    """Return the mapping observation values value owned by this checkpoint lifecycle module."""
    if isinstance(observation, dict):
        return True, extract_current_observation_from_dict(observation)
    if not hasattr(observation, "items"):
        return False, {}
    try:
        mapping = dict(observation.items())
    except (TypeError, ValueError) as exc:
        if quality_reporter is not None:
            quality_reporter(
                "checkpoint_observation_mapping_fallback",
                exc,
            )
        return False, {}
    return True, extract_current_observation_from_dict(mapping)


def _mixed_observation_values(flat: np.ndarray) -> dict[str, float]:
    """Return the mixed observation values value owned by this checkpoint lifecycle module."""
    values: dict[str, float] = {}
    offset = 0
    for name in POLICY_OBSERVATION_PARAM_NAMES:
        length = POLICY_OBSERVATION_PARAM_LENGTHS.get(name, 1)
        values[name] = float(flat[offset + length - 1])
        offset += length
    return values


def _sequential_observation_values(
    flat: np.ndarray,
) -> dict[str, float]:
    """Return the sequential observation values value owned by this checkpoint lifecycle module."""
    values: dict[str, float] = {}
    for index, value in enumerate(flat):
        column = (
            POLICY_OBSERVATION_PARAM_NAMES[index]
            if index < len(POLICY_OBSERVATION_PARAM_NAMES)
            else f"obs_{index}"
        )
        values[column] = float(value)
    return values


def _observation_to_row_values(
    observation: object,
    quality_reporter: QualityReporter | None,
) -> dict[str, float]:
    """Return the observation to row values value owned by this checkpoint lifecycle module."""
    is_mapping, mapping_values = _mapping_observation_values(
        observation,
        quality_reporter,
    )
    if is_mapping:
        return mapping_values
    array = np.asarray(observation, dtype=float)
    if array.size == 0:
        return {}
    flat = array.reshape(-1)
    expected_mixed_size = sum(
        POLICY_OBSERVATION_PARAM_LENGTHS.get(name, 1)
        for name in POLICY_OBSERVATION_PARAM_NAMES
    )
    if flat.size == expected_mixed_size:
        return _mixed_observation_values(flat)
    return _sequential_observation_values(flat)


def _observation_batches(observations: object) -> list[object]:
    """Return the observation batches value owned by this checkpoint lifecycle module."""
    if isinstance(observations, dict) or hasattr(observations, "items"):
        return [observations]
    array = np.asarray(observations, dtype=object)
    if array.size == 0:
        return []
    if array.ndim == 0:
        return [array.item()]
    if array.ndim == 1 and all(
        np.asarray(item).ndim == 0 for item in array
    ):
        return [array.reshape(-1)]
    return list(array)


def _normalized_observation_row(
    observation: object,
    *,
    env_index: int,
    total_timesteps: int,
    source: str,
    reward_values: list[float],
    info_values: list[object],
    quality_reporter: QualityReporter | None,
) -> dict[str, object] | None:
    """Return the normalized observation row value owned by this checkpoint lifecycle module."""
    observation_values = _observation_to_row_values(
        observation,
        quality_reporter,
    )
    if not observation_values:
        return None
    row: dict[str, object] = {
        "total_timesteps": int(total_timesteps or 0),
        "env_index": env_index,
        "source": source,
    }
    if env_index < len(reward_values):
        row["reward"] = reward_values[env_index]
    if env_index < len(info_values):
        row.update(_reward_component_fields(info_values[env_index]))
        row.update(_observation_diagnostic_fields(info_values[env_index]))
        row.update(_diagnostic_trace_fields(info_values[env_index]))
    row.update(observation_values)
    return row


def _normalize_observation_rows(
    observations: object,
    *,
    total_timesteps: int = 0,
    source: str = "step",
    rewards: object = None,
    infos: object = None,
    quality_reporter: QualityReporter | None = None,
) -> list[dict[str, object]]:
    """Normalize the observation rows accepted by this checkpoint lifecycle module."""
    if observations is None:
        return []
    reward_values = _reward_rows(rewards)
    info_values = list(infos or [])
    rows: list[dict[str, object]] = []
    for env_index, observation in enumerate(
        _observation_batches(observations)
    ):
        row = _normalized_observation_row(
            observation,
            env_index=env_index,
            total_timesteps=total_timesteps,
            source=source,
            reward_values=reward_values,
            info_values=info_values,
            quality_reporter=quality_reporter,
        )
        if row is not None:
            rows.append(row)
    return rows


def _checkpoint_debug_directory(checkpoint_path: str) -> str:
    """Return the directory that owns SB3 checkpoint metric summaries."""

    if os.path.isdir(str(checkpoint_path)):
        return str(checkpoint_path)
    return os.path.dirname(str(checkpoint_path))


def _metric_stat_rows(grouped_stats: dict[str, dict[str, float]]) -> list[dict]:
    """Return the metric stat rows value owned by this checkpoint lifecycle module."""
    rows = []
    for metric_name in sorted(grouped_stats.keys()):
        stats = grouped_stats[metric_name]
        min_value = float(stats.get("min", np.nan))
        max_value = float(stats.get("max", np.nan))
        p25 = float(stats.get("p25", np.nan))
        p75 = float(stats.get("p75", np.nan))
        rows.append({
            "metric": metric_name,
            "mean": float(stats.get("mean", np.nan)),
            "min": min_value,
            "max": max_value,
            "std": float(stats.get("std", np.nan)),
            "p25": p25,
            "p50": float(stats.get("p50", np.nan)),
            "p75": p75,
            "range": max_value - min_value if np.isfinite(min_value) and np.isfinite(max_value) else np.nan,
            "iqr": p75 - p25 if np.isfinite(p25) and np.isfinite(p75) else np.nan,
        })
    return rows


def _write_metric_stats_csv(
    checkpoint_path: str,
    filename: str,
    grouped_stats: dict,
    logger=None,
) -> str | None:
    """Persist the metric stats csv owned by this checkpoint lifecycle module."""
    rows = _metric_stat_rows(grouped_stats)
    if not rows:
        return None

    debug_directory = _checkpoint_debug_directory(checkpoint_path)
    os.makedirs(debug_directory, exist_ok=True)
    csv_path = os.path.join(debug_directory, filename)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHECKPOINT_METRIC_STATS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if logger is not None:
        logger.info(f"SB3 checkpoint metric stats CSV saved to {csv_path}")
    return csv_path


def write_checkpoint_metric_stats_csvs(
    checkpoint_path: str,
    metric_snapshots,
    logger=None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Persist the checkpoint metric stats csvs owned by this checkpoint lifecycle module."""
    snapshots = [
        snapshot
        for snapshot in list(metric_snapshots or [])
        if isinstance(snapshot, dict) and snapshot
    ]
    if not snapshots:
        return None, None, None, None

    observation_snapshots = [
        filtered
        for snapshot in snapshots
        if (filtered := filter_observation_stats(snapshot))
    ]
    reward_snapshots = [
        filtered
        for snapshot in snapshots
        if (filtered := filter_reward_stats(snapshot))
    ]
    episode_setup_snapshots = [
        filtered
        for snapshot in snapshots
        if (filtered := filter_episode_setup_stats(snapshot))
    ]
    episode_info_snapshots = [
        filtered
        for snapshot in snapshots
        if (filtered := filter_episode_info_stats(snapshot))
    ]
    observation_path = _write_metric_stats_csv(
        checkpoint_path,
        "epoch_observation_stats.csv",
        historical_stat_groups_from_snapshots(observation_snapshots),
        logger=logger,
    )
    reward_path = _write_metric_stats_csv(
        checkpoint_path,
        "epoch_reward_stats.csv",
        historical_stat_groups_from_snapshots(reward_snapshots),
        logger=logger,
    )
    episode_setup_path = _write_metric_stats_csv(
        checkpoint_path,
        "epoch_episode_setup_stats.csv",
        historical_stat_groups_from_snapshots(episode_setup_snapshots),
        logger=logger,
    )
    episode_info_path = _write_metric_stats_csv(
        checkpoint_path,
        "epoch_episode_info_stats.csv",
        historical_stat_groups_from_snapshots(episode_info_snapshots),
        logger=logger,
    )
    return observation_path, reward_path, episode_setup_path, episode_info_path


def _append_or_replace_observation_row(rows: list[dict], row: dict) -> None:
    """Append the or replace observation row owned by this checkpoint lifecycle module."""
    row_env_index = int(row.get("env_index", 0) or 0)
    row_total_timesteps = int(row.get("total_timesteps", 0) or 0)
    for index in range(len(rows) - 1, -1, -1):
        existing = rows[index]
        if int(existing.get("env_index", 0) or 0) != row_env_index:
            continue
        if int(existing.get("total_timesteps", 0) or 0) != row_total_timesteps:
            continue
        rows[index] = row
        return
    rows.append(row)

def save_sb3_checkpoint(
        *,
        model,
        checkpoint_directory: str,
        model_package_directory: str,
        scenario_index: int,
        raw_step: int,
        base_step_offset: int = 0,
        curriculum_state=None,
        parent_checkpoint_name: str | None = None,
        episode_observations=None,
        metric_snapshots=None,
        manual: bool = False,
        logger=None,
        checkpoint_repository: RuntimeCheckpointRepository | None = None,
) -> str:
    """Persist the sb3 checkpoint owned by this checkpoint lifecycle module."""
    step_num = effective_checkpoint_step(raw_step, base_step_offset)
    checkpoint_name = build_sb3_checkpoint_name(scenario_index, step_num, manual=manual)
    checkpoint_kind = "manual" if manual else "checkpoint"
    repository = checkpoint_repository or RuntimeCheckpointRepository(
        model_package_directory
    )
    with repository.mutation() as mutation:
        final_checkpoint_path = _reserve_sb3_checkpoint_path(
            checkpoint_directory,
            checkpoint_name,
        )
        checkpoint_path = mutation.prepare_checkpoint_target(
            final_checkpoint_path,
            artifact_kind=CheckpointArtifactKind.FILE_BUNDLE,
        )
        resolved_name = os.path.basename(final_checkpoint_path)
        model.save(checkpoint_path)
        mutation.write_checkpoint_manifest(
            checkpoint_path,
            checkpoint_name=resolved_name,
            checkpoint_kind=checkpoint_kind,
            framework_uuid="sb3",
            scenario_index=scenario_index,
            curriculum_state=curriculum_state,
            step_count=step_num,
            parent_checkpoint_name=parent_checkpoint_name,
        )
        try:
            write_checkpoint_observation_csv(
                checkpoint_path,
                episode_observations,
                logger=logger,
            )
        except _OPTIONAL_TELEMETRY_ERRORS as exc:
            _warn_quality_degradation(
                logger,
                "checkpoint_observation_csv_write_failed",
                exc,
            )
        try:
            write_checkpoint_metric_stats_csvs(
                checkpoint_path,
                metric_snapshots,
                logger=logger,
            )
        except _OPTIONAL_TELEMETRY_ERRORS as exc:
            _warn_quality_degradation(
                logger,
                "checkpoint_metric_csv_write_failed",
                exc,
            )
        update_last_checkpoint_metadata(
            mutation,
            model_package_directory,
            resolved_name,
            checkpoint_kind,
            logger=logger,
        )
    if logger is not None:
        logger.info(f"Model saved to {final_checkpoint_path}")
    return resolved_name


class StableBaselinesCheckpointMetadataCallback(BaseCallback):
    """Handle stable baselines checkpoint metadata lifecycle callbacks."""

    def __init__(
        self,
        *,
        scenario_manager,
        scenario_index_state,
        save_freq: int,
        checkpoint_directory: str,
        curriculum_index_state=None,
        parent_checkpoint_name: str | None = None,
        base_step_offset: int = 0,
        diagnostic_logger=None,
        model_package_directory: str | None = None,
        checkpoint_repository: RuntimeCheckpointRepository | None = None,
    ):
        """Initialize the stable baselines checkpoint metadata callback and its owned state."""
        super().__init__()
        self._scenario_manager = scenario_manager
        self._scenario_index_state = scenario_index_state
        self._curriculum_index_state = curriculum_index_state
        self._save_freq = int(save_freq)
        self._checkpoint_directory = checkpoint_directory
        self._model_package_directory = (
            model_package_directory or os.path.dirname(checkpoint_directory)
        )
        self._parent_checkpoint_name = parent_checkpoint_name
        self._base_step_offset = max(0, int(base_step_offset or 0))
        self._diagnostic_logger = diagnostic_logger
        self._checkpoint_repository = (
            checkpoint_repository
            or RuntimeCheckpointRepository(
                self._model_package_directory
            )
        )
        self._quality_degradations: set[str] = set()
        self._last_written_step = -1
        self._pending_observation_checkpoint_paths: list[str] = []
        self._pending_episode_observations: list[
            SB3CheckpointObservationRow
        ] = []
        self._pending_episode_env_index: int | None = None
        self._pending_capture_waiting_for_episode_boundary = False
        self._pending_diagnostic_trace_run_id: str | None = None
        self._historical_metric_snapshots: list[dict[str, float]] = []

    def _scenario_index(self) -> int:
        """Return the scenario index value owned by this stable baselines checkpoint metadata
        callback.
        """
        try:
            scenario_index = self._scenario_index_state.get()
        except Exception as exc:
            raise CheckpointStateReadError(
                "Checkpoint scenario state could not be read."
            ) from exc
        if type(scenario_index) is not int or scenario_index < 0:
            raise CheckpointStateReadError(
                "Checkpoint scenario state is invalid."
            )
        return scenario_index

    def _effective_step(self) -> int:
        """Return the effective step value owned by this stable baselines checkpoint metadata
        callback.
        """
        return effective_checkpoint_step(self.num_timesteps, self._base_step_offset)

    def _curriculum_state(self):
        """Return the curriculum state value owned by this stable baselines checkpoint metadata
        callback.
        """
        if self._curriculum_index_state is None:
            raise CheckpointStateReadError(
                "Checkpoint curriculum state is unavailable."
            )
        try:
            return self._curriculum_index_state.get()
        except Exception as exc:
            raise CheckpointStateReadError(
                "Checkpoint curriculum state could not be read."
            ) from exc

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return the quality degradations exposed by this stable baselines checkpoint metadata
        callback.
        """
        return frozenset(self._quality_degradations)

    def _record_quality_degradation(self, event: str, exc: BaseException) -> None:
        """Record the quality degradation owned by this stable baselines checkpoint metadata
        callback.
        """
        first_occurrence = event not in self._quality_degradations
        self._quality_degradations.add(event)
        if first_occurrence:
            _warn_quality_degradation(self._diagnostic_logger, event, exc)

    def _queue_next_episode_observation_capture(self, checkpoint_path: str) -> None:
        """Queue the next episode observation capture owned by this stable baselines checkpoint
        metadata callback.
        """
        self._pending_observation_checkpoint_paths.append(checkpoint_path)
        if len(self._pending_observation_checkpoint_paths) == 1:
            self._pending_episode_observations = []
            self._pending_episode_env_index = None
            self._pending_diagnostic_trace_run_id = uuid.uuid4().hex
            self._pending_capture_waiting_for_episode_boundary = not self._current_step_finished_episode()
            if not self._pending_capture_waiting_for_episode_boundary:
                self._arm_diagnostic_trace_capture(0)

    def _arm_diagnostic_trace_capture(self, env_index: int) -> None:
        """Best-effort arm the selected environment after a clean boundary."""

        try:
            self.training_env.env_method(
                "arm_diagnostic_trace_capture",
                indices=int(env_index),
            )
        except AttributeError:
            return
        except _OPTIONAL_TELEMETRY_ERRORS as exc:
            self._record_quality_degradation(
                "checkpoint_diagnostic_trace_arm_failed",
                exc,
            )

    def _complete_pending_episode_observation_capture(self) -> None:
        """Complete the pending episode observation capture for checkpoint metadata."""
        if not self._pending_observation_checkpoint_paths or not self._pending_episode_observations:
            return

        for checkpoint_path in self._pending_observation_checkpoint_paths:
            self._write_optional_checkpoint_telemetry(
                checkpoint_path,
                observation_rows=self._pending_episode_observations,
                metric_snapshots=self._historical_metric_snapshots,
            )
        self._pending_observation_checkpoint_paths.clear()
        self._pending_episode_observations = []
        self._pending_episode_env_index = None
        self._pending_capture_waiting_for_episode_boundary = False
        self._pending_diagnostic_trace_run_id = None

    def _apply_training_trace_identity(
        self,
        rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Attach one stable identity to explicitly projected trace rows."""

        trace_run_id = self._pending_diagnostic_trace_run_id
        if not trace_run_id:
            return rows
        identity = {
            "trace_run_id": trace_run_id,
            "framework": "sb3",
            "mode": "training",
        }
        for row in rows:
            declared = row.get("_diagnostic_trace_fields")
            if not isinstance(declared, tuple | list):
                continue
            row.update(identity)
            row["_diagnostic_trace_fields"] = tuple(
                dict.fromkeys((*identity, *(str(field) for field in declared)))
            )
        return rows

    def _write_optional_checkpoint_telemetry(
        self,
        checkpoint_path: str,
        *,
        observation_rows=None,
        metric_snapshots=None,
    ) -> None:
        """Persist the optional checkpoint telemetry owned by this stable baselines checkpoint
        metadata callback.
        """
        if observation_rows:
            try:
                write_checkpoint_observation_csv(checkpoint_path, observation_rows)
            except _OPTIONAL_TELEMETRY_ERRORS as exc:
                self._record_quality_degradation(
                    "checkpoint_observation_csv_write_failed",
                    exc,
                )
        if metric_snapshots:
            try:
                write_checkpoint_metric_stats_csvs(checkpoint_path, metric_snapshots)
            except _OPTIONAL_TELEMETRY_ERRORS as exc:
                self._record_quality_degradation(
                    "checkpoint_metric_csv_write_failed",
                    exc,
                )

    def _pending_env_matches(self, env_index: int) -> bool:
        """Return the pending env matches value owned by this stable baselines checkpoint metadata
        callback.
        """
        env_index = int(env_index or 0)
        if self._pending_episode_env_index is None:
            self._pending_episode_env_index = env_index
        return env_index == self._pending_episode_env_index

    def _current_step_finished_episode(self) -> bool:
        """Return the current step finished episode value owned by this stable baselines checkpoint
        metadata callback.
        """
        infos = self.locals.get("infos", [])
        return any(
            isinstance(info, dict) and (info.get("term") or info.get("trunc"))
            for info in infos
        )

    def _record_metric_snapshots(self) -> None:
        """Record the metric snapshots owned by this stable baselines checkpoint metadata callback.
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            if not (info.get("term") or info.get("trunc")):
                continue
            metrics = flatten_numeric_metrics(info.get("lanistai", {}))
            if metrics:
                self._historical_metric_snapshots.append(metrics)

    def _release_capture_at_episode_boundary(
        self,
        infos: list[object],
    ) -> None:
        """Release the capture at episode boundary owned by this stable baselines checkpoint
        metadata callback.
        """
        for env_index, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            if not (info.get("term") or info.get("trunc")):
                continue
            if self._pending_env_matches(env_index):
                self._pending_capture_waiting_for_episode_boundary = False
                self._pending_episode_observations = []
                self._arm_diagnostic_trace_capture(env_index)
                return

    def _normalize_pending_step_rows(
        self,
        observations: object,
        rewards: object,
        infos: list[object],
    ) -> list[dict[str, object]]:
        """Normalize the pending step rows accepted by this stable baselines checkpoint metadata
        callback.
        """
        try:
            return self._apply_training_trace_identity(
                _normalize_observation_rows(
                    observations,
                    total_timesteps=self._effective_step(),
                    source="step",
                    rewards=rewards,
                    infos=infos,
                    quality_reporter=self._record_quality_degradation,
                )
            )
        except (TypeError, ValueError) as exc:
            self._record_quality_degradation(
                "checkpoint_observation_normalization_failed",
                exc,
            )
            return []

    def _append_pending_step_rows(
        self,
        normalized_rows: list[dict[str, object]],
    ) -> None:
        """Append the pending step rows owned by this stable baselines checkpoint metadata
        callback.
        """
        for row in normalized_rows:
            env_index = int(row.get("env_index", 0))
            if not self._pending_env_matches(env_index):
                continue
            _append_or_replace_observation_row(
                self._pending_episode_observations,
                row,
            )

    def _normalize_terminal_rows(
        self,
        info: dict[str, object],
        *,
        env_index: int,
        reward_values: list[float],
    ) -> list[dict[str, object]]:
        """Normalize the terminal rows accepted by this stable baselines checkpoint metadata
        callback.
        """
        rewards = (
            [reward_values[env_index]]
            if env_index < len(reward_values)
            else None
        )
        try:
            return self._apply_training_trace_identity(
                _normalize_observation_rows(
                    info.get("terminal_observation"),
                    total_timesteps=self._effective_step(),
                    source="terminal",
                    rewards=rewards,
                    infos=[info],
                    quality_reporter=self._record_quality_degradation,
                )
            )
        except (TypeError, ValueError) as exc:
            self._record_quality_degradation(
                "checkpoint_terminal_observation_normalization_failed",
                exc,
            )
            return []

    def _record_terminal_rows(
        self,
        infos: list[object],
        reward_values: list[float],
    ) -> None:
        """Record the terminal rows owned by this stable baselines checkpoint metadata callback."""
        for env_index, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            if not self._pending_env_matches(env_index):
                continue
            if "terminal_observation" in info:
                terminal_rows = self._normalize_terminal_rows(
                    info,
                    env_index=env_index,
                    reward_values=reward_values,
                )
                for row in terminal_rows:
                    row["env_index"] = env_index
                    row["source"] = "terminal"
                    _append_or_replace_observation_row(
                        self._pending_episode_observations,
                        row,
                    )
            if info.get("term") or info.get("trunc"):
                self._complete_pending_episode_observation_capture()
                return

    def _record_pending_episode_observations(self) -> None:
        """Record the pending episode observations owned by this stable baselines checkpoint
        metadata callback.
        """
        if not self._pending_observation_checkpoint_paths:
            return
        infos = list(self.locals.get("infos", []))
        if self._pending_capture_waiting_for_episode_boundary:
            self._release_capture_at_episode_boundary(infos)
            return
        observations = self.locals.get("new_obs")
        rewards = self.locals.get("rewards")
        normalized_rows = self._normalize_pending_step_rows(
            observations,
            rewards,
            infos,
        )
        self._append_pending_step_rows(normalized_rows)
        self._record_terminal_rows(infos, _reward_rows(rewards))

    def episode_observations_for_checkpoint(
        self,
    ) -> list[SB3CheckpointObservationRow]:
        """Return a snapshot of the optional rows captured for manual saves."""

        return list(self._pending_episode_observations)

    def metric_snapshots_for_checkpoint(self) -> list[dict[str, float]]:
        """Return optional metric history for an explicit/manual save."""

        return list(self._historical_metric_snapshots)

    def _checkpoint_due(self) -> bool:
        """Checkpoint the due owned by this stable baselines checkpoint metadata callback."""
        return self._save_freq > 0 and self.n_calls % self._save_freq == 0

    def _on_step(self) -> bool:
        """Handle the step event for this stable baselines checkpoint metadata callback."""
        self._record_metric_snapshots()
        self._record_pending_episode_observations()
        if not self._checkpoint_due():
            return True
        effective_step = self._effective_step()
        if effective_step == self._last_written_step:
            return True

        scenario_index = self._scenario_index()
        curriculum_state = self._curriculum_state()
        with self._checkpoint_repository.mutation() as mutation:
            checkpoint_name = build_sb3_checkpoint_name(
                scenario_index,
                effective_step,
            )
            final_checkpoint_path = _reserve_sb3_checkpoint_path(
                self._checkpoint_directory,
                checkpoint_name,
            )
            checkpoint_path = mutation.prepare_checkpoint_target(
                final_checkpoint_path,
                artifact_kind=CheckpointArtifactKind.FILE_BUNDLE,
            )
            resolved_name = os.path.basename(final_checkpoint_path)
            self.model.save(checkpoint_path)
            mutation.write_checkpoint_manifest(
                checkpoint_path,
                checkpoint_name=resolved_name,
                checkpoint_kind="checkpoint",
                framework_uuid="sb3",
                scenario_index=scenario_index,
                curriculum_state=curriculum_state,
                step_count=effective_step,
                parent_checkpoint_name=self._parent_checkpoint_name,
            )
            self._write_optional_checkpoint_telemetry(
                checkpoint_path,
                metric_snapshots=self._historical_metric_snapshots,
            )
            update_last_checkpoint_metadata(
                mutation,
                self._model_package_directory,
                resolved_name,
                "checkpoint",
                logger=self._diagnostic_logger,
            )
        self._queue_next_episode_observation_capture(final_checkpoint_path)
        self._parent_checkpoint_name = resolved_name
        self._last_written_step = effective_step
        return True
