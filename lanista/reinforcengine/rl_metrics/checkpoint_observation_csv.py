# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Persist framework-neutral checkpoint episode observation traces."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping

from lanista.reinforcengine.rl_metrics.channel_schema import (
    POLICY_OBSERVATION_PARAM_NAMES,
)

CHECKPOINT_OBSERVATION_CSV_COLUMNS = [
    "total_timesteps",
    "reward",
    *POLICY_OBSERVATION_PARAM_NAMES,
    "tl_phase_remaining_delta_norm",
    "tl_phase_remaining_raw_delta_norm",
    "tl_timing_context_changed",
    "tl_phase_remaining_monotonic_violation",
    "reward_accel_normalized_stats_mean",
    "reward_accel_raw_stats_mean",
    "reward_accel_rolling_average_stats_mean",
    "reward_delta_normalized_stats_mean",
    "reward_delta_raw_stats_mean",
    "reward_delta_rolling_average_stats_mean",
    "reward_control_abs_supervised_cmd_accel_delta_change_stats_mean",
    "reward_control_abs_supervised_cmd_accel_delta_stats_mean",
    "reward_control_supervised_cmd_accel_discount_jitter_stats_mean",
    "reward_control_supervised_cmd_accel_delta_change_stats_mean",
    "reward_control_supervised_cmd_accel_delta_stats_mean",
    "reward_control_supervised_cmd_accel_local_jitter_stats_mean",
    "reward_control_supervised_cmd_accel_local_jitter_sign_change_stats_mean",
    "reward_control_supervised_cmd_accel_local_effective_jitter_stats_mean",
    "reward_control_supervised_cmd_accel_trend_jitter_stats_mean",
    "reward_control_supervised_cmd_accel_trend_jitter_sign_change_stats_mean",
    "reward_control_supervised_cmd_accel_trend_effective_jitter_stats_mean",
    "reward_control_supervised_cmd_accel_comfort_jitter_stats_mean",
    "reward_control_supervised_cmd_accel_comfort_jitter_sign_change_stats_mean",
    "reward_control_supervised_cmd_accel_comfort_effective_jitter_stats_mean",
    "reward_speed_normalized_stats_mean",
    "reward_speed_raw_stats_mean",
    "reward_sync_normalized_stats_mean",
    "reward_sync_raw_stats_mean",
    "reward_sync_rolling_average_stats_mean",
    "reward_total_raw_stats_mean",
]


def _checkpoint_debug_directory(checkpoint_path: str) -> str:
    """Return the directory that owns one checkpoint's diagnostic files."""

    if os.path.isdir(str(checkpoint_path)):
        return str(checkpoint_path)
    return os.path.dirname(str(checkpoint_path))


def write_checkpoint_observation_csv(
    checkpoint_path: str,
    observation_rows: Iterable[Mapping[str, object]] | None,
    logger: object | None = None,
) -> str | None:
    """Persist one shared checkpoint episode trace without framework coupling."""

    rows = [dict(row) for row in (observation_rows or ())]
    if not rows:
        return None

    debug_directory = _checkpoint_debug_directory(checkpoint_path)
    os.makedirs(debug_directory, exist_ok=True)
    csv_path = os.path.join(debug_directory, "episode_observations.csv")
    fieldnames = list(CHECKPOINT_OBSERVATION_CSV_COLUMNS)
    extra_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("reward_") and key not in fieldnames
        }
    )
    fieldnames.extend(extra_columns)
    diagnostic_columns: list[str] = []
    for row in rows:
        declared_fields = row.get("_diagnostic_trace_fields", ())
        if not isinstance(declared_fields, tuple | list):
            continue
        for fieldname in declared_fields:
            name = str(fieldname)
            if name not in diagnostic_columns:
                diagnostic_columns.append(name)
    fieldnames.extend(diagnostic_columns)

    seen_fieldnames: set[str] = set()
    fieldnames = [
        fieldname
        for fieldname in fieldnames
        if not (fieldname in seen_fieldnames or seen_fieldnames.add(fieldname))
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    info = getattr(logger, "info", None)
    if callable(info):
        info(f"Checkpoint observation CSV saved to {csv_path}")
    return csv_path


__all__ = [
    "CHECKPOINT_OBSERVATION_CSV_COLUMNS",
    "write_checkpoint_observation_csv",
]
