# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Pure numeric extraction, aggregation, grouping, and metric filtering."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable
from typing import Protocol, TypeGuard, cast, overload

import numpy as np

from lanista.reinforcengine.rl_metrics.channel_schema import (
    INTERACTION_EPISODE_METRIC_NAMES,
)


class _FloatConvertible(Protocol):
    """Define the typed boundary for float convertible."""

    def __float__(self) -> float:
        """Return the numeric value as a float."""


def is_valid_number(value: object) -> TypeGuard[_FloatConvertible]:
    """Return whether ``value`` follows the historical finite-number rule."""

    return isinstance(value, numbers.Number) and math.isfinite(
        float(cast(_FloatConvertible, value))
    )


@overload
def extract_current_observation_from_dict(
    observation: dict[str, object],
) -> dict[str, object]:
    """Return the extract current observation from dict value provided by this aggregation module.
    """
    ...


@overload
def extract_current_observation_from_dict(observation: object) -> object:
    """Return the extract current observation from dict value provided by this aggregation module.
    """
    ...


def extract_current_observation_from_dict(observation: object) -> object:
    """
    Extract current (latest) values from a time-windowed dictionary observation.

    Dictionary observations may contain history arrays or one-value scalar
    arrays. Metrics and CSV consumers need only the latest value, so extraction
    deliberately preserves the historical zero fallback for malformed channels.

    :param observation: Dictionary whose values are arrays or scalar values.
    :return: Dictionary containing one current scalar per observation key.
    """
    if not isinstance(observation, dict):
        return observation

    current_obs: dict[str, object] = {}
    for key, val in cast(dict[str, object], observation).items():
        if isinstance(val, list | tuple | np.ndarray):
            try:
                arr = np.asarray(val, dtype=float)
                current_obs[key] = arr[-1]
            except (IndexError, ValueError, TypeError):
                current_obs[key] = 0.0
        else:
            current_obs[key] = float(val) if isinstance(val, int | float) else 0.0

    return current_obs


def compute_stats(values: Iterable[float]) -> dict[str, float] | None:
    """Calculate the stable population-statistic set, or ``None`` when empty."""

    values = list(values)
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": p25,
        "p50": float(np.percentile(arr, 50)),
        "p75": p75,
        "min": min_value,
        "max": max_value,
        "range": max_value - min_value,
        "iqr": p75 - p25,
    }


def flatten_numeric_metrics(
    data: object,
    prefix: str = "",
    out: dict[str, float] | None = None,
) -> dict[str, float]:
    """Flatten numeric dictionary leaves depth-first while excluding booleans."""

    if out is None:
        out = {}
    if not isinstance(data, dict):
        return out

    for key, value in cast(dict[object, object], data).items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flatten_numeric_metrics(value, name, out)
        elif isinstance(value, bool):
            continue
        elif isinstance(value, numbers.Number):
            out[name] = float(cast(_FloatConvertible, value))
    return out


def group_stat_metrics(metrics: dict[str, float]) -> dict[str, dict[str, float]]:
    """Group dotted metric reductions without changing insertion order."""

    grouped: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        parts = key.split(".")
        if len(parts) < 2:
            continue
        stat_name = parts[-1]
        base_key = ".".join(parts[:-1])
        grouped.setdefault(base_key, {})[stat_name] = value
    return grouped


def historical_stat_groups_from_snapshots(
    snapshots: Iterable[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate finite historical snapshot values by dotted metric base key."""

    history: dict[str, list[float]] = {}
    for snapshot in snapshots:
        for key, value in snapshot.items():
            if not is_valid_number(value):
                continue
            parts = key.split(".")
            if len(parts) < 2:
                continue
            base_key = ".".join(parts[:-1])
            history.setdefault(base_key, []).append(float(value))

    grouped: dict[str, dict[str, float]] = {}
    for base_key, values in history.items():
        stats = compute_stats(values)
        if stats is not None:
            grouped[base_key] = stats
    return grouped


def filter_observation_stats(metrics: dict[str, float]) -> dict[str, float]:
    """Select observation distribution entries from a flattened snapshot."""

    return {
        key: value
        for key, value in metrics.items()
        if key.startswith("observations.") and ".stats." in key
    }


def filter_reward_stats(metrics: dict[str, float]) -> dict[str, float]:
    """Select reward distribution entries from a flattened snapshot."""

    return {
        key: value
        for key, value in metrics.items()
        if key.startswith("rewards.") and "_stats." in key
    }


def filter_episode_setup_stats(metrics: dict[str, float]) -> dict[str, float]:
    """Normalize current and legacy episode-setup scalars for aggregation."""

    filtered: dict[str, float] = {}
    for key, value in metrics.items():
        if not key.startswith("episode_setup."):
            continue
        if key.endswith(".value"):
            filtered[key] = value
            continue
        if ".stats." not in key:
            continue

        parts = key.split(".")
        if len(parts) < 4 or parts[-2] != "stats" or parts[-1] != "mean":
            continue

        # Older snapshots represented per-episode setup scalars as stats.mean.
        # Mapping only that reduction prevents reporting distributions of an
        # already aggregated scalar while retaining stored-run compatibility.
        filtered[".".join(parts[:-2] + ["value"])] = value
    return filtered


def filter_episode_info_stats(metrics: dict[str, float]) -> dict[str, float]:
    """Project binary events to rates and continuous episode data to values."""

    filtered: dict[str, float] = {}
    continuous_metric_keys = {
        "episode_info.terminated_reached_intersection",
        "episode_info.terminated_max_speed",
        "episode_info.terminated_average_speed",
        "episode_info.truncated_reached_intersection",
        "episode_info.truncated_max_speed",
        "episode_info.truncated_average_speed",
        *(f"episode_info.{name}" for name in INTERACTION_EPISODE_METRIC_NAMES),
    }
    for key, value in metrics.items():
        if not key.startswith("episode_info."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        reduction = parts[-1]
        if reduction != "mean":
            continue
        metric_key = ".".join(parts[:-1])
        if metric_key in continuous_metric_keys:
            filtered[f"{metric_key}.value"] = value
            continue
        # Episode-info metrics are binary indicators. Their per-episode sum and
        # mean are the same 0/1 value, so retaining both would create duplicate
        # historical rows and mislabel a rate as an accumulated count.
        filtered[f"{metric_key}.rate.value"] = value
    return filtered
