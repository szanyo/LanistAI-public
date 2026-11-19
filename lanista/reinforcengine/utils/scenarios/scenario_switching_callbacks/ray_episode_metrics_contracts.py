# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed, framework-neutral contracts for Ray episode metric processing.

The runtime store remains a plain dictionary because RLlib owns
``episode.custom_data``.  The typed view below documents every stable key while
keeping the serialized representation compatible with existing workers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypedDict

type MetricScalar = bool | int | float
type MetricSeries = list[MetricScalar]
type FloatSeriesMap = dict[str, list[float]]


class EpisodeMetricStore(TypedDict):
    """Stable per-episode metric payload stored below the ``lanistai`` key."""

    episode_terminated: MetricSeries
    episode_truncated: MetricSeries
    episode_end_conflict: MetricSeries
    episode_end_conflict_target_time_limit: MetricSeries
    episode_end_conflict_target_red_light_overdrive: MetricSeries
    curriculum_target_detected: MetricSeries
    curriculum_target_accepted: MetricSeries
    curriculum_target_reached: MetricSeries
    curriculum_target_rejected_by_red_light_overdrive: MetricSeries
    termination_curriculum_target: MetricSeries
    termination_route_end: MetricSeries
    truncation_time_limit: MetricSeries
    truncation_red_light_overdrive: MetricSeries
    unique_followed_vehicles: MetricSeries
    unique_blocking_pedestrians: MetricSeries
    lead_interaction_step_fraction: MetricSeries
    pedestrian_interaction_step_fraction: MetricSeries
    any_interaction_step_fraction: MetricSeries
    terminated_reached_intersection: MetricSeries
    terminated_max_speed: MetricSeries
    terminated_average_speed: MetricSeries
    truncated_reached_intersection: MetricSeries
    truncated_max_speed: MetricSeries
    truncated_average_speed: MetricSeries
    max_speed_per_episode: MetricSeries
    average_speed_per_episode: MetricSeries
    reached_intersection_number: MetricSeries
    number_of_vehicles: MetricSeries
    number_of_pedestrians: MetricSeries
    curriculum_metrics: FloatSeriesMap
    scenario_metrics: FloatSeriesMap
    telemetry_series: FloatSeriesMap
    diagnostic_series: FloatSeriesMap
    optional_telemetry_quality: FloatSeriesMap
    optional_telemetry_degradation_causes: list[str]
    sync_reward: MetricSeries
    normalized_sync_reward: MetricSeries
    avg_sync_penalty: MetricSeries
    accel_change_reward: MetricSeries
    normalized_accel_change_reward: MetricSeries
    avg_accel_penalty: MetricSeries
    avg_speed_reward: MetricSeries
    normalized_avg_speed_reward: MetricSeries
    delta_change_reward: MetricSeries
    normalized_delta_change_reward: MetricSeries
    avg_delta_penalty: MetricSeries
    observation_series: FloatSeriesMap
    reward_series: FloatSeriesMap
    episode_setup_series: FloatSeriesMap


class OptionalTelemetryQuality(StrEnum):
    """Fidelity of an optional observation, state, or telemetry read."""

    COMPLETE = "complete"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"

    @property
    def score(self) -> float:
        """Return the stable numeric quality score published to TensorBoard."""

        return {
            self.COMPLETE: 1.0,
            self.FALLBACK: 0.5,
            self.UNAVAILABLE: 0.0,
        }[self]


@dataclass(frozen=True, slots=True)
class OptionalStateRead:
    """One optional state read and its observable fidelity."""

    value: object | None
    quality: OptionalTelemetryQuality
    cause: str | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class OptionalTelemetryRead:
    """Finite diagnostic telemetry values and their observable fidelity."""

    values: dict[str, float]
    quality: OptionalTelemetryQuality
    cause: str | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class EpisodeStatus:
    """Episode completion values captured once at the start of a step hook."""

    overall_terminated: bool
    overall_truncated: bool
    is_terminated: bool
    is_truncated: bool
    is_done: bool

    @property
    def outcome_flags(self) -> tuple[bool, bool]:
        """Return one mutually exclusive termination/truncation outcome."""

        if self.overall_truncated:
            return False, True
        if self.overall_terminated:
            return True, False
        if self.is_truncated:
            return False, True
        if self.is_terminated:
            return True, False
        return False, False

    @property
    def terminated(self) -> bool:
        """Return the normalized successful terminal outcome."""

        return self.outcome_flags[0]

    @property
    def truncated(self) -> bool:
        """Return the normalized unsuccessful truncated outcome."""

        return self.outcome_flags[1]

    @property
    def requires_terminal_snapshot(self) -> bool:
        """Return whether terminal environment fields must be sampled now."""

        return (self.terminated or self.truncated) and self.is_done


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Normalized terminal flags derived from an environment snapshot."""

    terminated: bool
    truncated: bool
    termination_reason: object
    truncation_reason: object
    curriculum_target_detected: bool
    curriculum_target_accepted: bool
    curriculum_target_rejected_by_red_light_overdrive: bool


class EpisodeDataPort(Protocol):
    """Small episode surface shared by capture and publication components."""

    @property
    def custom_data(self) -> dict[str, object]:
        """Return mutable framework-owned custom episode data."""


class EpisodeStepPort(EpisodeDataPort, Protocol):
    """Episode state required by the step-capture pipeline."""

    @property
    def is_terminated(self) -> bool:
        """Return the framework's per-episode termination flag."""

    @property
    def is_truncated(self) -> bool:
        """Return the framework's per-episode truncation flag."""

    @property
    def is_done(self) -> bool:
        """Return whether the episode has completed all finalization work."""

    def get_terminateds(self) -> Mapping[str, object]:
        """Return agent and overall termination flags."""

    def get_truncateds(self) -> Mapping[str, object]:
        """Return agent and overall truncation flags."""

    def get_observations(self, *, indices: int, agent_ids: str) -> object:
        """Return the requested current agent observation."""


class MetricsLoggerPort(Protocol):
    """Narrow tuple-path metric logger used by the publisher."""

    def log_value(
        self,
        path: tuple[str, ...],
        value: object,
        *,
        reduce: str | None = None,
    ) -> None:
        """Publish one value with the requested framework reduction."""


type OptionalStateReader = Callable[[object | None], OptionalStateRead]


class EpisodeMetricsCapturePort(Protocol):
    """Ordered capture stages used by the RLlib hook adapter."""

    def initialize_episode(self, episode: EpisodeDataPort) -> None:
        """Install a new stable metric store on an episode."""

    def capture_status(self, episode: EpisodeStepPort) -> EpisodeStatus:
        """Capture completion flags once for the current step."""

    def capture_latest_observation(self, episode: EpisodeStepPort) -> None:
        """Append the latest policy observation values."""

    def select_environment(self, env: object | None, env_index: int) -> object | None:
        """Select and unwrap the indexed sub-environment."""

    def capture_step_series(self, episode: EpisodeDataPort, environment: object) -> None:
        """Append current reward and diagnostic telemetry values."""

    def capture_diagnostic_series(
        self,
        episode: EpisodeDataPort,
        environment: object,
        status: EpisodeStatus,
    ) -> None:
        """Append already-read semantic diagnostic values."""

    def capture_terminal_snapshot(
        self,
        episode: EpisodeDataPort,
        environment: object,
    ) -> None:
        """Append outcome, vehicle, curriculum, setup, and legacy values."""


class EpisodeMetricsPublisherPort(Protocol):
    """Ordered metric publication stages used by the RLlib hook adapter."""

    def initialize_environment_metrics(self, metrics_logger: MetricsLoggerPort) -> None:
        """Publish stable zero-valued keys before the first episode."""

    def publish_episode_scalars(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish canonical episode scalar paths and aliases."""

    def publish_curriculum_metrics(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish read-only curriculum state projections."""

    def publish_scenario_metrics(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish read-only scenario state projections."""

    def publish_intersection_metrics(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish episode-level intersection summaries."""

    def publish_observation_statistics(
        self,
        series: FloatSeriesMap,
        metrics_logger: MetricsLoggerPort,
        *,
        diagnostic_only: bool,
    ) -> None:
        """Publish compact statistics for observation or telemetry series."""

    def publish_episode_setup_metrics(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish the effective episode setup values."""

    def publish_reward_statistics(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish compact reward-component statistics."""

    def publish_diagnostic_statistics(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish compact semantic diagnostic statistics."""

    def publish_optional_telemetry_quality(
        self,
        store: EpisodeMetricStore,
        metrics_logger: MetricsLoggerPort,
    ) -> None:
        """Publish optional-source fidelity separately from domain values."""

    def record_episode_histograms(self, episode: EpisodeDataPort) -> None:
        """Append allowlisted finite histogram samples to the episode."""
