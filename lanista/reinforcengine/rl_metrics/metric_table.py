# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Deterministic human-readable formatting and logging of metric groups."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import numpy as np

from lanista.reinforcengine.rl_metrics.aggregation import is_valid_number

type MetricSortValue = float | str


class _MetricLogger(Protocol):
    """Define the typed boundary for metric logger."""

    def info(self, message: object) -> None:
        """Log one already-formatted metric-table line."""


def format_metric_table(
    prefix: str,
    grouped: dict[str, dict[str, float]],
    order: list[str] | None = None,
    sort_key: Callable[[str], MetricSortValue] | None = None,
    omit_uninformative_rows: bool = False,
) -> str:
    """Render a complete metric table as one newline-separated string."""

    return "\n".join(
        format_metric_table_lines(
            prefix,
            grouped,
            order=order,
            sort_key=sort_key,
            omit_uninformative_rows=omit_uninformative_rows,
        )
    )


def _metric_stat(stats: dict[str, float], key: str) -> float:
    """Return the metric stat value owned by this metric table module."""
    value = stats.get(key, float("nan"))
    return float(value) if is_valid_number(value) else float("nan")


def _metric_stats_are_all_zero(
    stats: dict[str, float],
    eps: float = 1e-12,
) -> bool:
    """Return the metric stats are all zero value owned by this metric table module."""
    min_value = _metric_stat(stats, "min")
    max_value = _metric_stat(stats, "max")
    mean_value = _metric_stat(stats, "mean")
    return (
        math.isfinite(min_value)
        and math.isfinite(max_value)
        and math.isfinite(mean_value)
        and abs(min_value) <= eps
        and abs(max_value) <= eps
        and abs(mean_value) <= eps
    )


def _metric_stats_are_absent_station_sentinel(
    metric: str,
    stats: dict[str, float],
    eps: float = 1e-9,
) -> bool:
    """Return the metric stats are absent station sentinel value owned by this metric table module.
    """
    if metric not in {
        "observations.station_1.stats",
        "observations.station_2.stats",
        "observations.station_3.stats",
    }:
        return False
    min_value = _metric_stat(stats, "min")
    max_value = _metric_stat(stats, "max")
    return (
        math.isfinite(min_value)
        and math.isfinite(max_value)
        and abs(min_value - (-200.0)) <= eps
        and abs(max_value - (-200.0)) <= eps
    )


def _metric_group_is_uninformative(
    metric: str,
    stats: dict[str, float],
) -> bool:
    """Return the metric group is uninformative value owned by this metric table module."""
    return _metric_stats_are_all_zero(
        stats
    ) or _metric_stats_are_absent_station_sentinel(metric, stats)


def compact_metric_groups(
    grouped: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Remove zero-only and absent-station sentinel groups from display."""

    return {
        key: stats
        for key, stats in grouped.items()
        if not _metric_group_is_uninformative(key, stats)
    }


def format_metric_table_lines(
    prefix: str,
    grouped: dict[str, dict[str, float]],
    order: list[str] | None = None,
    sort_key: Callable[[str], MetricSortValue] | None = None,
    omit_uninformative_rows: bool = False,
) -> list[str]:
    """Render deterministic metric-table lines for direct or streamed output."""

    if omit_uninformative_rows:
        grouped = compact_metric_groups(grouped)

    if not grouped:
        return []

    metric_width = max(len("metric"), max(len(key) for key in grouped))
    column_width = 8
    separator = "  "
    columns: list[tuple[str, Callable[[dict[str, float]], float]]] = [
        ("mean", lambda s: s.get("mean", float("nan"))),
        ("min", lambda s: s.get("min", float("nan"))),
        ("max", lambda s: s.get("max", float("nan"))),
        ("std", lambda s: s.get("std", float("nan"))),
        ("p25", lambda s: s.get("p25", float("nan"))),
        ("p50", lambda s: s.get("p50", float("nan"))),
        ("p75", lambda s: s.get("p75", float("nan"))),
        (
            "range",
            lambda s: (
                s.get("max", float("nan")) - s.get("min", float("nan"))
                if np.isfinite(s.get("min", float("nan")))
                and np.isfinite(s.get("max", float("nan")))
                else float("nan")
            ),
        ),
        (
            "iqr",
            lambda s: (
                s.get("p75", float("nan")) - s.get("p25", float("nan"))
                if np.isfinite(s.get("p25", float("nan")))
                and np.isfinite(s.get("p75", float("nan")))
                else float("nan")
            ),
        ),
    ]

    header_cols = separator.join(
        label.rjust(column_width) for label, _ in columns
    )
    lines = [prefix, f"{'metric'.ljust(metric_width)}{separator}{header_cols}"]

    if order:
        ordered_keys = [key for key in order if key in grouped]
        ordered_keys += [key for key in grouped if key not in order]
    else:
        ordered_keys = sorted(grouped.keys(), key=sort_key)

    for base_key in ordered_keys:
        stats = grouped[base_key]
        values = [fmt(stats) for _, fmt in columns]
        row_cols = separator.join(
            f"{value:{column_width}.4f}" for value in values
        )
        lines.append(f"{base_key.ljust(metric_width)}{separator}{row_cols}")

    return lines


def log_metric_table(
    logger: _MetricLogger,
    prefix: str,
    grouped: dict[str, dict[str, float]],
    order: list[str] | None = None,
    sort_key: Callable[[str], MetricSortValue] | None = None,
    omit_uninformative_rows: bool = False,
) -> None:
    """Emit a metric table one line per log record to preserve readability."""

    for line in format_metric_table_lines(
        prefix,
        grouped,
        order=order,
        sort_key=sort_key,
        omit_uninformative_rows=omit_uninformative_rows,
    ):
        logger.info(line)
