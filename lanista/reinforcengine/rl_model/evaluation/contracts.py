# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Define explicit evaluation scope and sweep lifecycle records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from pydantic.types import JsonValue
from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
    validate_library_id,
    validate_utc_timestamp,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class _FrozenEvaluationContract(BaseModel):
    """Reject unknown state and mutation in persisted evaluation records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SingleScenarioEvaluationScope(_FrozenEvaluationContract):
    """Select the existing interactive evaluation path by scenario index."""

    kind: Literal["single_scenario"] = "single_scenario"
    scenario_index: NonNegativeInt
    repetitions: PositiveInt = 1


class LibraryEvaluationScope(_FrozenEvaluationContract):
    """Select one immutable held-out library for complete ordered traversal."""

    kind: Literal["library"] = "library"
    library_id: str = Field(min_length=1)
    manifest_sha256: Sha256
    repetitions_per_scenario: PositiveInt = 1

    @field_validator("library_id")
    @classmethod
    def _validate_library_id(cls, value: str) -> str:
        return validate_library_id(value)


EvaluationScope = Annotated[
    SingleScenarioEvaluationScope | LibraryEvaluationScope,
    Field(discriminator="kind"),
]
_EVALUATION_SCOPE_ADAPTER: TypeAdapter[EvaluationScope] = TypeAdapter(EvaluationScope)


def validate_evaluation_scope(payload: object) -> EvaluationScope:
    """Validate one explicit single-scenario or library selection."""

    return _EVALUATION_SCOPE_ADAPTER.validate_python(payload)


class EvaluationSweepStatus(StrEnum):
    """Persistent terminal and non-terminal sweep states."""

    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ScenarioExecutionStatus(StrEnum):
    """Persistent result states for one attempted library scenario."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RunConfigurationSnapshot(_FrozenEvaluationContract):
    """Freeze the effective configuration used by one grouped-library run."""

    schema_version: Literal[1] = 1
    model_id: str = Field(min_length=1)
    framework: Literal["ray", "sb3"]
    session_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    checkpoint_sha256: Sha256
    library_id: str = Field(min_length=1)
    library_manifest_sha256: Sha256
    deterministic_inference: bool
    repetitions_per_scenario: PositiveInt
    headless_pacing: dict[str, JsonValue]
    effective_runtime_configuration: dict[str, JsonValue]
    software_versions: dict[str, str]

    @field_validator("library_id")
    @classmethod
    def _validate_library_id(cls, value: str) -> str:
        return validate_library_id(value)



class EvaluationGroupProgress(_FrozenEvaluationContract):
    """Counts for one named group, derived from persisted scenario records."""

    group_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    expected_scenario_count: PositiveInt
    completed_scenario_count: NonNegativeInt
    failed_scenario_count: NonNegativeInt
    cancelled_scenario_count: NonNegativeInt
    remaining_scenario_count: NonNegativeInt

    @model_validator(mode="after")
    def _validate_counts(self) -> EvaluationGroupProgress:
        total = (
            self.completed_scenario_count
            + self.failed_scenario_count
            + self.cancelled_scenario_count
            + self.remaining_scenario_count
        )
        if total != self.expected_scenario_count:
            raise ValueError("Group progress must account for every scenario.")
        return self


class EvaluationSweepProgress(_FrozenEvaluationContract):
    """Whole-run and per-group counts after a durable scenario attempt."""

    run_id: str = Field(min_length=1)
    expected_scenario_count: PositiveInt
    completed_scenario_count: NonNegativeInt
    failed_scenario_count: NonNegativeInt
    cancelled_scenario_count: NonNegativeInt
    remaining_scenario_count: NonNegativeInt
    groups: tuple[EvaluationGroupProgress, ...]

    @model_validator(mode="after")
    def _validate_counts(self) -> EvaluationSweepProgress:
        total = (
            self.completed_scenario_count
            + self.failed_scenario_count
            + self.cancelled_scenario_count
            + self.remaining_scenario_count
        )
        if total != self.expected_scenario_count:
            raise ValueError("Sweep progress must account for every scenario.")
        if self.groups and sum(
            group.expected_scenario_count for group in self.groups
        ) != self.expected_scenario_count:
            raise ValueError("Group progress must cover the whole sweep.")
        return self

class EvaluationScenarioExecutionRecord(_FrozenEvaluationContract):
    """Record one scenario attempt and its auditable evidence paths."""

    schema_version: Literal[1]
    sequence_index: NonNegativeInt
    scenario_id: str
    requested_vehicle_count: NonNegativeInt
    requested_pedestrian_count: NonNegativeInt
    replicate_index: NonNegativeInt
    seed: NonNegativeInt
    status: ScenarioExecutionStatus
    started_at: str
    ended_at: str
    evidence_files: tuple[str, ...]
    terminal_outcome: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    @field_validator("scenario_id")
    @classmethod
    def _validate_scenario_id(cls, value: str) -> str:
        return validate_library_id(value)

    @field_validator("started_at", "ended_at")
    @classmethod
    def _validate_record_timestamp(cls, value: str) -> str:
        return validate_utc_timestamp(value)

    @field_validator("evidence_files")
    @classmethod
    def _validate_evidence_files(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            path = PurePosixPath(value)
            if (
                not value
                or value != path.as_posix()
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
                or "\\" in value
            ):
                raise ValueError("Evidence paths must be normalized relative POSIX paths.")
            normalized.append(value)
        if tuple(normalized) != tuple(sorted(set(normalized))):
            raise ValueError("Evidence paths must be unique and canonically ordered.")
        return tuple(normalized)

    @model_validator(mode="after")
    def _validate_result_state(self) -> EvaluationScenarioExecutionRecord:
        started = datetime.strptime(self.started_at, "%Y-%m-%dT%H:%M:%SZ")
        ended = datetime.strptime(self.ended_at, "%Y-%m-%dT%H:%M:%SZ")
        if ended < started:
            raise ValueError("Scenario end timestamp cannot precede its start timestamp.")
        failure_type = str(self.failure_type or "").strip()
        failure_message = str(self.failure_message or "").strip()
        terminal_outcome = str(self.terminal_outcome or "").strip()
        if self.status is ScenarioExecutionStatus.FAILED:
            if not failure_type or not failure_message:
                raise ValueError("A failed scenario requires failure metadata.")
            if terminal_outcome:
                raise ValueError("A failed scenario cannot carry a terminal outcome.")
        elif failure_type or failure_message:
            raise ValueError("Only a failed scenario may carry failure metadata.")
        if self.status is ScenarioExecutionStatus.COMPLETED and not terminal_outcome:
            raise ValueError("A completed scenario requires a terminal outcome.")
        if self.status is ScenarioExecutionStatus.CANCELLED and terminal_outcome:
            raise ValueError("A cancelled scenario cannot carry a terminal outcome.")
        return self


class EvaluationSweepDescriptor(_FrozenEvaluationContract):
    """Record one checkpoint's progress over one immutable library."""

    schema_version: Literal[1]
    run_id: str = Field(min_length=1)
    framework: Literal["ray", "sb3"]
    model_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    checkpoint_sha256: Sha256
    library_id: str = Field(min_length=1)
    library_manifest_sha256: Sha256
    run_configuration_sha256: Sha256 | None = None
    deterministic_inference: bool
    repetitions_per_scenario: PositiveInt
    started_at: str = Field(min_length=1)
    ended_at: str | None = None
    status: EvaluationSweepStatus
    expected_scenario_count: PositiveInt
    completed_scenario_count: NonNegativeInt
    failed_scenario_count: NonNegativeInt
    software_versions: dict[str, str]
    failure_message: str | None = None

    @field_validator("library_id")
    @classmethod
    def _validate_library_id(cls, value: str) -> str:
        return validate_library_id(value)

    @field_validator("started_at", "ended_at")
    @classmethod
    def _validate_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_utc_timestamp(value)

    @model_validator(mode="after")
    def _validate_progress_state(self) -> EvaluationSweepDescriptor:
        attempted = self.completed_scenario_count + self.failed_scenario_count
        if attempted > self.expected_scenario_count:
            raise ValueError("Sweep progress exceeds the expected scenario count.")
        if self.status is EvaluationSweepStatus.COMPLETED and (
            self.completed_scenario_count != self.expected_scenario_count
            or self.failed_scenario_count != 0
        ):
            raise ValueError("A sweep cannot be marked completed until every scenario succeeds.")
        if self.status is EvaluationSweepStatus.RUNNING and self.ended_at is not None:
            raise ValueError("A running sweep cannot have an end timestamp.")
        if self.status is not EvaluationSweepStatus.RUNNING and self.ended_at is None:
            raise ValueError("A terminal sweep requires an end timestamp.")
        if self.ended_at is not None:
            started_at = datetime.strptime(self.started_at, "%Y-%m-%dT%H:%M:%SZ")
            ended_at = datetime.strptime(self.ended_at, "%Y-%m-%dT%H:%M:%SZ")
            if ended_at < started_at:
                raise ValueError("Sweep end timestamp cannot precede its start timestamp.")
        normalized_failure = str(self.failure_message or "").strip()
        if self.status is EvaluationSweepStatus.FAILED and not normalized_failure:
            raise ValueError("A failed sweep requires failure metadata.")
        if self.status is not EvaluationSweepStatus.FAILED and normalized_failure:
            raise ValueError("Only a failed sweep may carry failure metadata.")
        return self


__all__ = [
    "EvaluationScenarioExecutionRecord",
    "EvaluationGroupProgress",
    "EvaluationScope",
    "EvaluationSweepDescriptor",
    "EvaluationSweepProgress",
    "EvaluationSweepStatus",
    "LibraryEvaluationScope",
    "RunConfigurationSnapshot",
    "ScenarioExecutionStatus",
    "SingleScenarioEvaluationScope",
    "validate_evaluation_scope",
]
