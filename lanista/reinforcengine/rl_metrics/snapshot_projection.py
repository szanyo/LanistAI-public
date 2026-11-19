# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Project observations and environment state into stable metric snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import numpy as np

from lanista.reinforcengine.rl_metrics.aggregation import (
    extract_current_observation_from_dict,
    is_valid_number,
)
from lanista.reinforcengine.rl_metrics.channel_schema import (
    DIAGNOSTIC_OBSERVATION_PARAM_NAMES,
    OBSERVATION_PARAM_LENGTHS,
    OBSERVATION_PARAM_NAMES,
    POLICY_OBSERVATION_PARAM_LENGTHS,
    POLICY_OBSERVATION_PARAM_NAMES,
    REWARD_COMPONENT_SERIES,
    REWARD_SERIES_KEYS,
    TELEMETRY_OBSERVATION_PARAM_NAMES,
    TRAINING_DIAGNOSTIC_PARAM_NAMES,
)
from lanista.reinforcengine.rl_metrics.env_snapshot import EnvSnapshotDependencies
from lanista.reinforcengine.rl_metrics.env_snapshot import (
    build_env_lanistai_snapshot as _build_env_lanistai_snapshot,
)

type MetricSnapshot = dict[str, object]


class _FloatConvertible(Protocol):
    """Define the typed boundary for float convertible."""

    def __float__(self) -> float:
        """Return the numeric value as a float."""


class _TrainingDiagnosticGetter(Protocol):
    """Return one already-read training diagnostic mapping."""

    def __call__(
        self,
        *,
        terminated: bool,
        truncated: bool,
    ) -> Mapping[str, object]:
        """Project current diagnostics without simulator reads."""


def _set_nested_metric(
    root: MetricSnapshot,
    path_parts: tuple[str, ...],
    value: float,
) -> None:
    """Set the nested metric owned by this snapshot projection module."""
    cursor = root
    for part in path_parts[:-1]:
        cursor = cast(dict[str, object], cursor.setdefault(part, {}))
    cursor[path_parts[-1]] = float(value)


def _observation_stats_from_names(
    observation: object,
    param_names: Sequence[str],
    param_lengths: Mapping[str, int],
) -> MetricSnapshot:
    """Return the observation stats from names value owned by this snapshot projection module."""
    stats: MetricSnapshot = {}

    if isinstance(observation, dict):
        current_values: dict[str, object] = {}
        for key, val in cast(dict[str, object], observation).items():
            if isinstance(val, list | tuple | np.ndarray):
                try:
                    current_values[key] = float(np.asarray(val, dtype=float)[-1])
                except (IndexError, ValueError, TypeError):
                    current_values[key] = 0.0
            else:
                current_values[key] = (
                    float(val) if isinstance(val, int | float) else 0.0
                )
        for param_name in param_names:
            if param_name in current_values and is_valid_number(
                current_values[param_name]
            ):
                _set_nested_metric(
                    stats,
                    ("observations", param_name, "stats", "mean"),
                    float(cast(_FloatConvertible, current_values[param_name])),
                )
    else:
        # Flat Ray observations can contain history blocks followed by scalar
        # channels; old scalar-only layouts remain readable for checkpoints.
        obs_arr = np.asarray(observation, dtype=float).reshape(-1)
        offset = 0
        legacy_scalar_flat = obs_arr.size == len(param_names)
        for name in param_names:
            length = 1 if legacy_scalar_flat else param_lengths.get(name, 1)
            if offset >= len(obs_arr):
                break
            value_index = min(offset + length - 1, len(obs_arr) - 1)
            value = obs_arr[value_index]
            if is_valid_number(value):
                _set_nested_metric(
                    stats,
                    ("observations", name, "stats", "mean"),
                    float(value),
                )
            offset += length

    return stats


def observation_stats_from_observation(observation: object) -> MetricSnapshot:
    """Project every telemetry observation into the historical stats tree."""

    return _observation_stats_from_names(
        observation,
        OBSERVATION_PARAM_NAMES,
        OBSERVATION_PARAM_LENGTHS,
    )


def policy_observation_stats_from_observation(
    observation: object,
) -> MetricSnapshot:
    """Project only policy-visible observations into the stats tree."""

    return _observation_stats_from_names(
        observation,
        POLICY_OBSERVATION_PARAM_NAMES,
        POLICY_OBSERVATION_PARAM_LENGTHS,
    )


def telemetry_stats_from_observation(observation: object) -> MetricSnapshot:
    """Project the complete telemetry channel into the observation stats tree."""

    return _observation_stats_from_names(
        observation,
        TELEMETRY_OBSERVATION_PARAM_NAMES,
        OBSERVATION_PARAM_LENGTHS,
    )


def observation_channel_values_from_observation(
    observation: object,
) -> dict[str, dict[str, float]]:
    """Split current dictionary values into policy and telemetry channels."""

    if not isinstance(observation, dict):
        return {"observation": {}, "telemetry": {}}

    current_observation = extract_current_observation_from_dict(observation)
    return {
        "observation": {
            name: float(cast(_FloatConvertible, current_observation[name]))
            for name in POLICY_OBSERVATION_PARAM_NAMES
            if name in current_observation
            and is_valid_number(current_observation[name])
        },
        "telemetry": {
            name: float(cast(_FloatConvertible, current_observation[name]))
            for name in TELEMETRY_OBSERVATION_PARAM_NAMES
            if name in current_observation
            and is_valid_number(current_observation[name])
        },
    }


def reward_series_values_from_env(
    env: object,
    reward: object | None = None,
) -> dict[str, float]:
    """Read finite reward components from the typed getter or legacy fields."""

    values: dict[str, float] = {}
    if reward is not None:
        values["total"] = float(cast(_FloatConvertible, reward))

    reward_components_getter = getattr(env, "get_reward_component_series", None)
    if callable(reward_components_getter):
        getter = cast(Callable[[], Mapping[str, object]], reward_components_getter)
        for key, value in getter().items():
            if is_valid_number(value):
                values[key] = float(value)
        return values

    for key, spec in REWARD_COMPONENT_SERIES.items():
        value = getattr(env, spec["attr"], None)
        if is_valid_number(value):
            values[key] = float(value)
    return values


def reward_channel_values_from_env(
    env: object,
    reward: object | None = None,
) -> dict[str, float]:
    """Return the current reward-channel values for snapshot composition."""

    return reward_series_values_from_env(env, reward)


def reward_snapshot_from_env(env: object, reward: object) -> MetricSnapshot:
    """Project known reward series into their canonical nested metric paths."""

    snapshot: MetricSnapshot = {"rewards": {}}
    for key, value in reward_series_values_from_env(env, reward).items():
        path = REWARD_SERIES_KEYS.get(key)
        if path is None:
            continue
        _set_nested_metric(snapshot, path + ("mean",), float(value))
    return snapshot


def _overlay_training_diagnostics(
    snapshot: MetricSnapshot,
    env: object,
    *,
    terminated: bool,
    truncated: bool,
) -> None:
    """Overlay finite diagnostics through the existing step metric tree."""

    getter_candidate = getattr(
        env,
        "get_training_diagnostic_component_series",
        None,
    )
    if not callable(getter_candidate):
        return
    getter = cast(_TrainingDiagnosticGetter, getter_candidate)
    try:
        values = getter(terminated=terminated, truncated=truncated)
    except Exception:
        return
    for name in TRAINING_DIAGNOSTIC_PARAM_NAMES:
        value = values.get(name)
        if is_valid_number(value):
            _set_nested_metric(
                snapshot,
                ("diagnostics", name, "stats", "mean"),
                float(cast(_FloatConvertible, value)),
            )


_ENV_SNAPSHOT_DEPENDENCIES = EnvSnapshotDependencies(
    reward_snapshot=reward_snapshot_from_env,
    observation_snapshot=observation_stats_from_observation,
    observation_channels=observation_channel_values_from_observation,
    reward_channels=reward_channel_values_from_env,
    is_valid_number=is_valid_number,
    diagnostic_observation_names=frozenset(DIAGNOSTIC_OBSERVATION_PARAM_NAMES),
)


def build_env_lanistai_snapshot(
    env: object,
    observation: object,
    reward: object,
    terminated: bool,
    truncated: bool,
) -> MetricSnapshot:
    """Build one environment metric snapshot through the typed workflow."""

    snapshot = _build_env_lanistai_snapshot(
        env,
        observation,
        reward,
        terminated,
        truncated,
        dependencies=_ENV_SNAPSHOT_DEPENDENCIES,
    )
    _overlay_training_diagnostics(
        snapshot,
        env,
        terminated=terminated,
        truncated=truncated,
    )
    return snapshot
