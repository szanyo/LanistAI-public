# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Define scenario-level and demand-cell evaluation report rows."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(allow_inf_nan=False, ge=0)]
Rate = Annotated[float, Field(allow_inf_nan=False, ge=0, le=1)]


class _FrozenSummaryContract(BaseModel):
    """Reject unknown or non-finite report data at the serialization boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ScenarioExecutionStatus(StrEnum):
    """Outcome of executing one scenario evidence scope."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScenarioSummary(_FrozenSummaryContract):
    """Represent exactly one attempted scenario as the statistical base unit."""

    schema_version: Literal[1]
    scenario_id: str = Field(min_length=1)
    requested_vehicle_count: NonNegativeInt
    requested_pedestrian_count: NonNegativeInt
    replicate_index: NonNegativeInt
    seed: NonNegativeInt
    execution_status: ScenarioExecutionStatus
    terminal_outcome: str | None
    trip_completed: bool | None
    failure_message: str | None
    descriptor_relative_path: str = Field(min_length=1)
    trace_relative_path: str | None
    episode_duration_seconds: NonNegativeFloat | None
    completed_trip_duration_seconds: NonNegativeFloat | None
    stop_time_seconds: NonNegativeFloat | None
    stop_time_fraction: Rate | None
    mean_speed_mps: NonNegativeFloat | None
    mean_abs_applied_acceleration_mps2: NonNegativeFloat | None
    realised_acceleration_rms_mps2: NonNegativeFloat | None
    local_jitter_mean: FiniteFloat | None
    trend_jitter_mean: FiniteFloat | None
    comfort_jitter_mean: FiniteFloat | None
    optimizer_intervention_count: NonNegativeInt | None
    supervisor_intervention_count: NonNegativeInt | None
    control_deviation_mean: FiniteFloat | None
    red_light_event: bool | None
    vehicle_conflict_event: bool | None
    pedestrian_conflict_event: bool | None

    @field_validator("descriptor_relative_path", "trace_relative_path")
    @classmethod
    def _validate_evidence_path(cls, value: str | None) -> str | None:
        """Require portable run-relative evidence paths."""

        if value is None:
            return None
        path = PurePosixPath(value)
        if (
            not value
            or value != path.as_posix()
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in value
        ):
            raise ValueError("Evidence paths must be normalized relative POSIX paths.")
        return value

    @model_validator(mode="after")
    def _validate_status_coherence(self) -> ScenarioSummary:
        """Keep execution status, failure, and completed-trip evidence coherent."""

        failure_message = str(self.failure_message or "").strip()
        if self.execution_status is ScenarioExecutionStatus.FAILED and not failure_message:
            raise ValueError("A failed scenario requires failure metadata.")
        if self.execution_status is not ScenarioExecutionStatus.FAILED and failure_message:
            raise ValueError("Only a failed scenario may carry failure metadata.")
        if self.execution_status is ScenarioExecutionStatus.COMPLETED:
            if self.terminal_outcome is None or self.trace_relative_path is None:
                raise ValueError("A completed scenario requires terminal and trace evidence.")
        elif self.trip_completed:
            raise ValueError("A non-completed execution cannot complete its trip.")
        if self.completed_trip_duration_seconds is not None and self.trip_completed is not True:
            raise ValueError("Completed-trip duration requires a completed trip.")
        return self


class DescriptiveStatistics(_FrozenSummaryContract):
    """Describe one finite metric over independent scenario rows."""

    count: PositiveInt
    mean: FiniteFloat
    standard_deviation: NonNegativeFloat
    median: FiniteFloat
    interquartile_range: NonNegativeFloat
    q05: FiniteFloat
    q25: FiniteFloat
    q75: FiniteFloat
    q95: FiniteFloat

    @model_validator(mode="after")
    def _validate_distribution(self) -> DescriptiveStatistics:
        """Require ordered quantiles and a consistent interquartile range."""

        if not self.q05 <= self.q25 <= self.median <= self.q75 <= self.q95:
            raise ValueError("Descriptive-statistic quantiles must be ordered.")
        if not math.isclose(
            self.interquartile_range,
            self.q75 - self.q25,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Interquartile range must equal q75 minus q25.")
        return self


class GridSummary(_FrozenSummaryContract):
    """Aggregate one requested vehicle-by-pedestrian demand cell."""

    schema_version: Literal[1]
    requested_vehicle_count: NonNegativeInt
    requested_pedestrian_count: NonNegativeInt
    expected_scenario_count: PositiveInt
    attempted_scenario_count: NonNegativeInt
    completed_execution_count: NonNegativeInt
    failed_execution_count: NonNegativeInt
    completed_trip_count: NonNegativeInt
    outcome_counts: dict[str, NonNegativeInt]
    outcome_rates: dict[str, Rate]
    metrics: dict[str, DescriptiveStatistics]

    @model_validator(mode="after")
    def _validate_denominators(self) -> GridSummary:
        """Require counts, rates, and metric samples to share valid denominators."""

        if (
            self.attempted_scenario_count
            != self.completed_execution_count + self.failed_execution_count
            or self.attempted_scenario_count > self.expected_scenario_count
            or self.completed_trip_count > self.completed_execution_count
            or sum(self.outcome_counts.values()) != self.attempted_scenario_count
        ):
            raise ValueError("Grid summary denominators are inconsistent.")
        if set(self.outcome_counts) != set(self.outcome_rates):
            raise ValueError("Outcome count and rate categories must match.")
        for metric_name, statistics in self.metrics.items():
            denominator = (
                self.completed_trip_count
                if metric_name == "completed_trip_duration_seconds"
                else self.attempted_scenario_count
            )
            if statistics.count > denominator:
                raise ValueError("Metric count exceeds its scenario denominator.")
        if self.attempted_scenario_count:
            for outcome, count in self.outcome_counts.items():
                expected_rate = count / self.attempted_scenario_count
                if not math.isclose(
                    self.outcome_rates[outcome],
                    expected_rate,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("Outcome rates must use attempted scenarios as denominator.")
        elif any(rate != 0.0 for rate in self.outcome_rates.values()):
            raise ValueError("Zero-attempt outcome rates must be zero.")
        return self


class GroupedScenarioSummary(ScenarioSummary):
    "Add stable group provenance to one version 2 scenario observation."

    schema_version: Literal[2]
    group_id: str = Field(min_length=1)
    group_name: str = Field(min_length=1)
    realised_vehicle_count: NonNegativeInt
    realised_pedestrian_count: NonNegativeInt


class _GroupedAggregateSummary(_FrozenSummaryContract):
    "Validate one group or whole-library aggregate over scenario rows."

    schema_version: Literal[2]
    expected_scenario_count: PositiveInt
    attempted_scenario_count: NonNegativeInt
    completed_execution_count: NonNegativeInt
    failed_execution_count: NonNegativeInt
    completed_trip_count: NonNegativeInt
    outcome_counts: dict[str, NonNegativeInt]
    outcome_rates: dict[str, Rate]
    metrics: dict[str, DescriptiveStatistics]

    @model_validator(mode="after")
    def _validate_scenario_denominators(self) -> _GroupedAggregateSummary:
        "Require every count and rate to use scenarios as observations."

        attempted = self.attempted_scenario_count
        if (
            attempted > self.expected_scenario_count
            or attempted != self.completed_execution_count + self.failed_execution_count
            or self.completed_trip_count > self.completed_execution_count
            or sum(self.outcome_counts.values()) != attempted
            or set(self.outcome_counts) != set(self.outcome_rates)
        ):
            raise ValueError("Grouped summary denominators are inconsistent.")
        for metric_name, summary in self.metrics.items():
            maximum = (
                self.completed_trip_count
                if metric_name == "completed_trip_duration_seconds"
                else attempted
            )
            if summary.count > maximum:
                raise ValueError("Metric count exceeds its scenario denominator.")
        for outcome, count in self.outcome_counts.items():
            expected_rate = count / attempted if attempted else 0.0
            if not math.isclose(
                self.outcome_rates[outcome], expected_rate, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("Outcome rate must use attempted scenarios as denominator.")
        return self


class GroupSummary(_GroupedAggregateSummary):
    "Describe one named library group using its own scenario observations."

    group_id: str = Field(min_length=1)
    group_name: str = Field(min_length=1)
    requested_vehicle_count: NonNegativeInt
    requested_pedestrian_count: NonNegativeInt


class LibrarySummary(_GroupedAggregateSummary):
    "Describe the full library using every scenario at equal weight."

    library_id: str = Field(min_length=1)


__all__ = [
    "DescriptiveStatistics",
    "GroupSummary",
    "GridSummary",
    "GroupedScenarioSummary",
    "LibrarySummary",
    "ScenarioExecutionStatus",
    "ScenarioSummary",
]
