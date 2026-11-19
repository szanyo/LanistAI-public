# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared metric-table reporting for rolling-statistics adapters."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from lanista.reinforcengine.rl_metrics.aggregation import (
    filter_episode_info_stats,
    filter_episode_setup_stats,
    filter_observation_stats,
    filter_reward_stats,
    historical_stat_groups_from_snapshots,
)
from lanista.reinforcengine.rl_metrics.channel_schema import (
    EPISODE_INFO_STAT_ORDER,
    EPISODE_SETUP_STAT_ORDER,
    REWARD_STAT_ORDER,
)
from lanista.reinforcengine.rl_metrics.metric_table import log_metric_table

type MetricSnapshot = dict[str, float]
type MetricGroups = dict[str, dict[str, float]]
type SnapshotFilter = Callable[[MetricSnapshot], MetricSnapshot]


class MetricLogger(Protocol):
    """Minimal logging surface required by metric tables."""

    def info(self, message: object) -> None:
        """Log one already-formatted metric-table line."""


class MetricReportingPort(Protocol):
    """Adapter state consumed by the shared reporting method."""

    _logger: MetricLogger
    _windowed_metric_snapshots: deque[MetricSnapshot]
    _historical_metric_snapshots: list[MetricSnapshot]
    _OBSERVATION_ORDER: list[str]
    _METRIC_WINDOW_SCOPE: str


@dataclass(frozen=True, slots=True)
class GroupedMetricSnapshots:
    """Historical metric groups for the four stable logging tables."""

    observations: MetricGroups
    rewards: MetricGroups
    episode_setup: MetricGroups
    episode_info: MetricGroups


@dataclass(frozen=True, slots=True)
class MetricSnapshotReporter:
    """Report one complete metric window and retain long-term history."""

    observation_order: tuple[str, ...]
    scope: str = "native_sample"

    def log_and_clear(
        self,
        *,
        logger: MetricLogger,
        windowed_snapshots: deque[MetricSnapshot],
        historical_snapshots: Sequence[MetricSnapshot],
    ) -> None:
        """Log one complete native-sample window, then clear that window."""
        _ = historical_snapshots
        if not windowed_snapshots:
            return
        window_capacity = windowed_snapshots.maxlen
        if (
            window_capacity is not None
            and len(windowed_snapshots) < window_capacity
        ):
            return

        snapshots = list(windowed_snapshots)
        groups = _group_snapshots(snapshots)
        heading = (
            f"[RollingStats.metric_window.{{group}}] "
            f"scope={self.scope} samples={len(snapshots)}"
        )
        log_metric_table(
            logger,
            heading.format(group="observation_stats"),
            groups.observations,
            order=list(self.observation_order),
            omit_uninformative_rows=True,
        )
        log_metric_table(
            logger,
            heading.format(group="reward_stats"),
            _ordered_groups(groups.rewards, REWARD_STAT_ORDER),
            order=REWARD_STAT_ORDER,
            omit_uninformative_rows=True,
        )
        log_metric_table(
            logger,
            heading.format(group="episode_setup_stats"),
            groups.episode_setup,
            order=EPISODE_SETUP_STAT_ORDER,
        )
        log_metric_table(
            logger,
            heading.format(group="episode_info_stats"),
            groups.episode_info,
            order=EPISODE_INFO_STAT_ORDER,
            omit_uninformative_rows=True,
        )
        windowed_snapshots.clear()


def log_window_metric_stats(port: MetricReportingPort) -> None:
    """Bound adapter method backed by the shared reporting collaborator."""

    history = cast(
        Sequence[MetricSnapshot],
        getattr(port, "_historical_metric_snapshots", ()),
    )
    reporter = MetricSnapshotReporter(
        tuple(port._OBSERVATION_ORDER),
        scope=getattr(port, "_METRIC_WINDOW_SCOPE", "native_sample"),
    )
    reporter.log_and_clear(
        logger=port._logger,
        windowed_snapshots=port._windowed_metric_snapshots,
        historical_snapshots=history,
    )


def _ordered_groups(
    groups: MetricGroups,
    order: Sequence[str],
) -> MetricGroups:
    """Keep only canonical console rows while preserving their display order."""

    return {name: groups[name] for name in order if name in groups}


def _group_snapshots(
    snapshots: Sequence[MetricSnapshot],
) -> GroupedMetricSnapshots:
    """Return the group snapshots value owned by this rolling stats reporting module."""
    return GroupedMetricSnapshots(
        observations=historical_stat_groups_from_snapshots(
            _filtered_snapshots(snapshots, filter_observation_stats)
        ),
        rewards=historical_stat_groups_from_snapshots(
            _filtered_snapshots(snapshots, filter_reward_stats)
        ),
        episode_setup=historical_stat_groups_from_snapshots(
            _filtered_snapshots(snapshots, filter_episode_setup_stats)
        ),
        episode_info=historical_stat_groups_from_snapshots(
            _filtered_snapshots(snapshots, filter_episode_info_stats)
        ),
    )


def _filtered_snapshots(
    snapshots: Sequence[MetricSnapshot],
    snapshot_filter: SnapshotFilter,
) -> list[MetricSnapshot]:
    """Return the filtered snapshots value owned by this rolling stats reporting module."""
    filtered_snapshots = []
    for snapshot in snapshots:
        filtered = snapshot_filter(snapshot)
        if filtered:
            filtered_snapshots.append(filtered)
    return filtered_snapshots
