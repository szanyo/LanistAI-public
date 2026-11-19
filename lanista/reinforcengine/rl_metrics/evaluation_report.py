# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Reduce a terminal evaluation sweep to stable scenario and grid reports."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from lanista.reinforcengine.rl_metrics.evaluation_grouped_publication import (
    publish_grouped_report,
    read_grouped_run_context,
)
from lanista.reinforcengine.rl_metrics.evaluation_summary_contracts import (
    DescriptiveStatistics,
    GroupSummary,
    GridSummary,
    GroupedScenarioSummary,
    LibrarySummary,
    ScenarioSummary,
)
from lanista.reinforcengine.rl_metrics.evaluation_summary_contracts import (
    ScenarioExecutionStatus as SummaryExecutionStatus,
)
from lanista.reinforcengine.rl_model.evaluation.contracts import (
    EvaluationScenarioExecutionRecord,
    EvaluationSweepDescriptor,
    EvaluationSweepStatus,
    ScenarioExecutionStatus,
)
from lanista.reinforcengine.rl_model.evaluation.coordinator import (
    EVALUATION_LIBRARY_RUN_FILENAME,
    SCENARIO_RESULT_FILENAME,
)

if TYPE_CHECKING:
    from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
        GroupedEvaluationLibraryManifest,
    )

SCENARIO_SUMMARY_FILENAME = "scenario-summary.csv"
GROUP_SUMMARY_FILENAME = "group-summary.csv"
GRID_SUMMARY_FILENAME = "grid-summary.csv"
LIBRARY_SUMMARY_FILENAME = "evaluation-library-summary.json"
GROUPED_LIBRARY_SUMMARY_FILENAME = "library-summary.json"
_TRACE_FILENAME = "evaluation_trace.csv"
_EVALUATION_DESCRIPTOR_FILENAME = "evaluation-run.json"
_METRICS = (
    "episode_duration_seconds",
    "completed_trip_duration_seconds",
    "stop_time_seconds",
    "stop_time_fraction",
    "mean_speed_mps",
    "mean_abs_applied_acceleration_mps2",
    "realised_acceleration_rms_mps2",
    "local_jitter_mean",
    "trend_jitter_mean",
    "comfort_jitter_mean",
    "optimizer_intervention_count",
    "supervisor_intervention_count",
    "control_deviation_mean",
    "red_light_event",
    "vehicle_conflict_event",
    "pedestrian_conflict_event",
)
_TRACE_ALIASES = {
    "local_jitter_mean": (
        "reward_control_supervised_cmd_accel_local_jitter_stats_mean",
        "reward_control_supervised_cmd_accel_local_jitter",
    ),
    "trend_jitter_mean": (
        "reward_control_supervised_cmd_accel_trend_jitter_stats_mean",
        "reward_control_supervised_cmd_accel_trend_jitter",
    ),
    "comfort_jitter_mean": (
        "reward_control_supervised_cmd_accel_weighted_comfort_jitter_stats_mean",
        "reward_control_supervised_cmd_accel_weighted_comfort_jitter",
        "reward_control_supervised_cmd_accel_comfort_jitter_stats_mean",
        "reward_control_supervised_cmd_accel_comfort_jitter",
    ),
}
_TIME_FIELDS = ("simulation_time_seconds", "episode_time_seconds", "elapsed_time_seconds")


@dataclass(slots=True)
class _Diagnostics:
    """Accumulate invalid trace values while scenarios are reduced."""

    invalid_values: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class EvaluationReportResult:
    """Return the stable report location and its scenario-level reductions."""

    summary_root: Path
    scenarios: tuple[ScenarioSummary, ...]
    grid: tuple[GridSummary, ...]
    groups: tuple[GroupSummary, ...] = ()
    library: LibrarySummary | None = None


def _read_model[ModelT: (EvaluationSweepDescriptor, EvaluationScenarioExecutionRecord)](
    path: Path,
    model_type: type[ModelT],
) -> ModelT:
    """Read and validate one persisted evaluation contract."""

    try:
        return model_type.model_validate_json(path.read_bytes())
    except Exception as error:
        raise RuntimeError(f"Invalid evaluation evidence: {path!s}.") from error


def _finite(row: Mapping[str, str], field_name: str, diagnostics: _Diagnostics) -> float | None:
    """Return one finite trace value while counting malformed data."""

    raw = str(row.get(field_name, "") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        diagnostics.invalid_values[field_name] += 1
        return None
    if not math.isfinite(value):
        diagnostics.invalid_values[field_name] += 1
        return None
    return value


def _values(rows: Iterable[Mapping[str, str]], name: str, diagnostics: _Diagnostics) -> list[float]:
    """Collect the finite values for one trace field."""

    return [value for row in rows if (value := _finite(row, name, diagnostics)) is not None]


def _alias_values(
    rows: tuple[dict[str, str], ...], aliases: tuple[str, ...], diagnostics: _Diagnostics
) -> list[float]:
    """Collect values from the first trace alias represented by the schema."""

    field_name = next((name for name in aliases if any(name in row for row in rows)), None)
    return _values(rows, field_name, diagnostics) if field_name is not None else []


def _mean(values: Iterable[float]) -> float | None:
    """Return the arithmetic mean or preserve a missing metric."""

    finite = tuple(values)
    return statistics.fmean(finite) if finite else None


def _event(rows: tuple[dict[str, str], ...], name: str, diagnostics: _Diagnostics) -> bool | None:
    """Reduce one recorded indicator to a scenario-level event flag."""

    values = _values(rows, name, diagnostics)
    return any(value != 0.0 for value in values) if values else None


def _trace_outcome(
    rows: tuple[dict[str, str], ...], fallback: str | None
) -> tuple[str | None, bool | None]:
    """Resolve terminal reasons and whether every episode reached route end."""

    reasons = sorted(
        {
            str(row.get("termination_reason") or row.get("truncation_reason") or "").strip()
            for row in rows
            if str(row.get("termination_reason") or row.get("truncation_reason") or "").strip()
        }
    )
    if not reasons:
        return fallback, None
    outcome = reasons[0] if len(reasons) == 1 else "mixed:" + "|".join(reasons)
    return outcome, all(reason == "route_end" for reason in reasons)


def _recorded_step_duration(
    rows: tuple[dict[str, str], ...], diagnostics: _Diagnostics
) -> float | None:
    """Sum configured simulator step lengths per episode when time is not logged."""

    if not any("step_length_seconds" in row for row in rows):
        return None
    by_episode: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row, "step_length_seconds", diagnostics)
        if value is not None:
            if value > 0:
                by_episode[str(row.get("episode_index", "0"))].append(value)
            else:
                diagnostics.invalid_values["step_length_seconds"] += 1
    return _mean(sum(values) for values in by_episode.values() if values)


def _duration(rows: tuple[dict[str, str], ...], diagnostics: _Diagnostics) -> float | None:
    """Prefer explicit timestamps, then recorded configured step lengths."""

    field_name = next((name for name in _TIME_FIELDS if any(name in row for row in rows)), None)
    if field_name is None:
        return _recorded_step_duration(rows, diagnostics)
    by_episode: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row, field_name, diagnostics)
        if value is not None:
            by_episode[str(row.get("episode_index", "0"))].append(value)
    durations = [max(values) - min(values) for values in by_episode.values() if values]
    return _mean(durations)


def _find_evidence(root: Path, filename: str, *, required: bool) -> Path | None:
    """Resolve one regular non-symlink evidence file below a scenario root."""

    matches = sorted(
        path for path in root.rglob(filename) if path.is_file() and not path.is_symlink()
    )
    if len(matches) > 1 or (required and not matches):
        raise RuntimeError(f"Expected exactly one {filename} below {root!s}.")
    return matches[0] if matches else None


def _read_trace(path: Path) -> tuple[dict[str, str], ...]:
    """Read a non-empty evaluation trace with an unambiguous header."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RuntimeError(f"Invalid evaluation trace header: {path!s}.")
        rows = tuple(dict(row) for row in reader)
    if not rows:
        raise RuntimeError(f"Evaluation trace is empty: {path!s}.")
    return rows


def _require_declared_evidence(
    evidence_root: Path,
    record: EvaluationScenarioExecutionRecord,
    path: Path | None,
) -> None:
    """Require discovered evidence to belong to the scenario's closed inventory."""

    if path is not None and path.relative_to(evidence_root).as_posix() not in record.evidence_files:
        raise RuntimeError("Discovered file is absent from the scenario evidence inventory.")


def _paired_commands(
    rows: tuple[dict[str, str], ...], diagnostics: _Diagnostics
) -> tuple[tuple[float, float], ...]:
    """Pair finite applied and supervised commands without hiding diagnostics."""

    pairs: list[tuple[float, float]] = []
    untracked = _Diagnostics()
    for row in rows:
        applied_value = _finite(row, "applied_command_mps2", untracked)
        supervised_value = _finite(row, "supervised_command_mps2", diagnostics)
        if applied_value is not None and supervised_value is not None:
            pairs.append((applied_value, supervised_value))
    return tuple(pairs)


def _scenario_summary(
    run_root: Path,
    record: EvaluationScenarioExecutionRecord,
    diagnostics: _Diagnostics,
) -> ScenarioSummary:
    """Reduce one scenario attempt and its trace to one statistical row."""

    evidence_root = run_root / "scenarios" / record.scenario_id
    completed = record.status is ScenarioExecutionStatus.COMPLETED
    descriptor = _find_evidence(evidence_root, _EVALUATION_DESCRIPTOR_FILENAME, required=completed)
    trace = _find_evidence(evidence_root, _TRACE_FILENAME, required=completed)
    _require_declared_evidence(evidence_root, record, descriptor)
    _require_declared_evidence(evidence_root, record, trace)
    rows = _read_trace(trace) if trace is not None else ()
    outcome, trip_completed = _trace_outcome(rows, record.terminal_outcome)
    duration = _duration(rows, diagnostics)
    speeds = _values(rows, "actual_speed_mps", diagnostics)
    applied = _values(rows, "applied_command_mps2", diagnostics)
    accelerations = _values(rows, "actual_acceleration_mps2", diagnostics)
    stopped = _values(rows, "motion_state_stopped", diagnostics)
    paired_commands = _paired_commands(rows, diagnostics)
    stop_fraction = _mean(stopped)
    return ScenarioSummary(
        schema_version=1,
        scenario_id=record.scenario_id,
        requested_vehicle_count=record.requested_vehicle_count,
        requested_pedestrian_count=record.requested_pedestrian_count,
        replicate_index=record.replicate_index,
        seed=record.seed,
        execution_status=SummaryExecutionStatus(record.status.value),
        terminal_outcome=outcome if completed else None,
        trip_completed=trip_completed if completed else None,
        failure_message=record.failure_message,
        descriptor_relative_path=(descriptor or evidence_root / SCENARIO_RESULT_FILENAME)
        .relative_to(run_root)
        .as_posix(),
        trace_relative_path=trace.relative_to(run_root).as_posix() if trace else None,
        episode_duration_seconds=duration,
        completed_trip_duration_seconds=duration if trip_completed is True else None,
        stop_time_seconds=duration * stop_fraction
        if duration is not None and stop_fraction is not None
        else None,
        stop_time_fraction=stop_fraction,
        mean_speed_mps=_mean(speeds),
        mean_abs_applied_acceleration_mps2=_mean(abs(value) for value in applied),
        realised_acceleration_rms_mps2=(
            math.sqrt(statistics.fmean(value * value for value in accelerations))
            if accelerations
            else None
        ),
        local_jitter_mean=_mean(
            _alias_values(rows, _TRACE_ALIASES["local_jitter_mean"], diagnostics)
        ),
        trend_jitter_mean=_mean(
            _alias_values(rows, _TRACE_ALIASES["trend_jitter_mean"], diagnostics)
        ),
        comfort_jitter_mean=_mean(
            _alias_values(rows, _TRACE_ALIASES["comfort_jitter_mean"], diagnostics)
        ),
        optimizer_intervention_count=(
            sum(
                value != 0.0
                for value in _values(rows, "use_case_optimizer_intervention", diagnostics)
            )
            if rows and any("use_case_optimizer_intervention" in row for row in rows)
            else None
        ),
        supervisor_intervention_count=(
            sum(value != 0.0 for value in _values(rows, "use_case_sumo_intervention", diagnostics))
            if rows and any("use_case_sumo_intervention" in row for row in rows)
            else None
        ),
        control_deviation_mean=_mean(abs(a - s) for a, s in paired_commands),
        red_light_event=_event(rows, "red_light_overdrive", diagnostics),
        vehicle_conflict_event=_event(rows, "supervisor_vehicle_conflict", diagnostics),
        pedestrian_conflict_event=_event(rows, "supervisor_pedestrian_conflict", diagnostics),
    )


def _quantile(values: tuple[float, ...], probability: float) -> float:
    """Interpolate one deterministic quantile from finite scenario values."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _statistics(values: Iterable[float]) -> DescriptiveStatistics | None:
    """Describe one metric over independent scenario observations."""

    finite = tuple(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    q25 = _quantile(finite, 0.25)
    q75 = _quantile(finite, 0.75)
    return DescriptiveStatistics(
        count=len(finite),
        mean=statistics.fmean(finite),
        standard_deviation=statistics.stdev(finite) if len(finite) > 1 else 0.0,
        median=statistics.median(finite),
        interquartile_range=q75 - q25,
        q05=_quantile(finite, 0.05),
        q25=q25,
        q75=q75,
        q95=_quantile(finite, 0.95),
    )


def _grid_summary(rows: tuple[ScenarioSummary, ...]) -> tuple[GridSummary, ...]:
    """Aggregate scenario rows by requested vehicle and pedestrian counts."""

    grouped: dict[tuple[int, int], list[ScenarioSummary]] = defaultdict(list)
    for row in rows:
        grouped[(row.requested_vehicle_count, row.requested_pedestrian_count)].append(row)
    summaries: list[GridSummary] = []
    for (vehicle_count, pedestrian_count), cell in sorted(grouped.items()):
        outcomes = Counter(
            "execution_failed"
            if row.execution_status.value == "failed"
            else str(row.terminal_outcome or row.execution_status.value)
            for row in cell
        )
        attempted = len(cell)
        metric_statistics = {
            metric: summary
            for metric in _METRICS
            if (
                summary := _statistics(
                    float(value) for row in cell if (value := getattr(row, metric)) is not None
                )
            )
            is not None
        }
        summaries.append(
            GridSummary(
                schema_version=1,
                requested_vehicle_count=vehicle_count,
                requested_pedestrian_count=pedestrian_count,
                expected_scenario_count=attempted,
                attempted_scenario_count=attempted,
                completed_execution_count=sum(
                    row.execution_status.value == "completed" for row in cell
                ),
                failed_execution_count=sum(row.execution_status.value == "failed" for row in cell),
                completed_trip_count=sum(row.trip_completed is True for row in cell),
                outcome_counts=dict(sorted(outcomes.items())),
                outcome_rates={name: count / attempted for name, count in sorted(outcomes.items())},
                metrics=metric_statistics,
            )
        )
    return tuple(summaries)


def _grouped_aggregate_fields(
    rows: tuple[GroupedScenarioSummary, ...],
    *,
    expected_count: int,
) -> dict[str, object]:
    "Reduce one exact scenario population to common aggregate fields."

    attempted = len(rows)
    if attempted != expected_count:
        raise RuntimeError("Grouped report requires every declared scenario.")
    outcomes = Counter(
        "execution_failed"
        if row.execution_status is SummaryExecutionStatus.FAILED
        else str(row.terminal_outcome or row.execution_status.value)
        for row in rows
    )
    metrics = {
        metric: summary
        for metric in _METRICS
        if (
            summary := _statistics(
                float(value) for row in rows if (value := getattr(row, metric)) is not None
            )
        )
        is not None
    }
    return {
        "expected_scenario_count": expected_count,
        "attempted_scenario_count": attempted,
        "completed_execution_count": sum(
            row.execution_status is SummaryExecutionStatus.COMPLETED for row in rows
        ),
        "failed_execution_count": sum(
            row.execution_status is SummaryExecutionStatus.FAILED for row in rows
        ),
        "completed_trip_count": sum(row.trip_completed is True for row in rows),
        "outcome_counts": dict(sorted(outcomes.items())),
        "outcome_rates": {
            name: count / attempted for name, count in sorted(outcomes.items())
        },
        "metrics": metrics,
    }


def _grouped_summaries(
    rows: tuple[GroupedScenarioSummary, ...],
    manifest: GroupedEvaluationLibraryManifest,
) -> tuple[tuple[GroupSummary, ...], LibrarySummary]:
    "Aggregate each named group and the library from the same scenario rows."

    group_summaries: list[GroupSummary] = []
    known_group_ids = {group.group_id for group in manifest.groups}
    if {row.group_id for row in rows} != known_group_ids:
        raise RuntimeError("Scenario group identity does not match the library manifest.")
    for group in manifest.groups:
        members = tuple(row for row in rows if row.group_id == group.group_id)
        if any(
            row.group_name != group.name
            or row.requested_vehicle_count != group.vehicle_count
            or row.requested_pedestrian_count != group.pedestrian_count
            for row in members
        ):
            raise RuntimeError("Scenario group factors do not match the library manifest.")
        group_summaries.append(
            GroupSummary(
                schema_version=2,
                group_id=group.group_id,
                group_name=group.name,
                requested_vehicle_count=group.vehicle_count,
                requested_pedestrian_count=group.pedestrian_count,
                **_grouped_aggregate_fields(members, expected_count=group.scenario_count),
            )
        )
    library = LibrarySummary(
        schema_version=2,
        library_id=manifest.library_id,
        **_grouped_aggregate_fields(
            rows,
            expected_count=manifest.expected_scenario_count,
        ),
    )
    return tuple(group_summaries), library


def _csv_value(value: object) -> object:
    """Project missing values and booleans into stable CSV scalars."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Atomically replace one report artifact with flushed bytes."""

    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_bytes(fieldnames: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> bytes:
    """Serialize ordered mappings as deterministic UTF-8 CSV bytes."""

    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    return output.getvalue().encode("utf-8")


def _grid_csv_rows(
    grid: tuple[GridSummary | GroupSummary, ...],
) -> tuple[dict[str, object], ...]:
    """Flatten grid or group statistics into stable publication columns."""

    rows: list[dict[str, object]] = []
    for summary in grid:
        row = summary.model_dump(exclude={"outcome_counts", "outcome_rates", "metrics"})
        row["outcome_counts_json"] = json.dumps(
            summary.outcome_counts, sort_keys=True, separators=(",", ":")
        )
        row["outcome_rates_json"] = json.dumps(
            summary.outcome_rates, sort_keys=True, separators=(",", ":")
        )
        for metric in _METRICS:
            stats = summary.metrics.get(metric)
            for statistic in (
                "count",
                "mean",
                "standard_deviation",
                "median",
                "interquartile_range",
                "q05",
                "q25",
                "q75",
                "q95",
            ):
                row[f"{metric}__{statistic}"] = getattr(stats, statistic) if stats else None
        rows.append(row)
    return tuple(rows)


def _read_scenario_records(
    root: Path,
    descriptor: EvaluationSweepDescriptor,
) -> tuple[EvaluationScenarioExecutionRecord, ...]:
    """Restore and cross-check every scenario record against its sweep."""

    record_paths = sorted((root / "scenarios").glob(f"*/{SCENARIO_RESULT_FILENAME}"))
    record_items = tuple(
        (path, _read_model(path, EvaluationScenarioExecutionRecord)) for path in record_paths
    )
    if any(
        path.is_symlink() or path.parent.is_symlink() or path.parent.name != record.scenario_id
        for path, record in record_items
    ):
        raise RuntimeError("Scenario result does not match its evidence directory.")
    if len(record_items) != descriptor.expected_scenario_count:
        raise RuntimeError("Scenario-result count does not match the sweep descriptor.")
    records = tuple(
        sorted(
            (record for _, record in record_items),
            key=lambda record: record.sequence_index,
        )
    )
    if tuple(record.sequence_index for record in records) != tuple(range(len(records))):
        raise RuntimeError("Scenario-result sequence is incomplete or duplicated.")
    completed_count = sum(record.status is ScenarioExecutionStatus.COMPLETED for record in records)
    failed_count = sum(record.status is ScenarioExecutionStatus.FAILED for record in records)
    if (
        completed_count != descriptor.completed_scenario_count
        or failed_count != descriptor.failed_scenario_count
    ):
        raise RuntimeError("Scenario status counts do not match the sweep descriptor.")
    return records


def _grouped_report(
    root: Path,
    descriptor: EvaluationSweepDescriptor,
) -> EvaluationReportResult:
    """Reduce one verified completed grouped run and publish all four outputs."""

    manifest = read_grouped_run_context(root, descriptor)
    records = _read_scenario_records(root, descriptor)
    if any(
        record.status is not ScenarioExecutionStatus.COMPLETED
        or record.scenario_id != entry.scenario_id
        or record.requested_vehicle_count != entry.requested_vehicle_count
        or record.requested_pedestrian_count != entry.requested_pedestrian_count
        or record.replicate_index != entry.replicate_index
        or record.seed != entry.seed
        for record, entry in zip(records, manifest.scenarios, strict=True)
    ):
        raise RuntimeError("Scenario records do not match the frozen library manifest.")

    diagnostics = _Diagnostics()
    group_names = {group.group_id: group.name for group in manifest.groups}
    scenarios = tuple(
        GroupedScenarioSummary.model_validate(
            {
                **_scenario_summary(root, record, diagnostics).model_dump(),
                "schema_version": 2,
                "group_id": entry.group_id,
                "group_name": group_names[entry.group_id],
                "realised_vehicle_count": entry.realised_vehicle_count,
                "realised_pedestrian_count": entry.realised_pedestrian_count,
            }
        )
        for record, entry in zip(records, manifest.scenarios, strict=True)
    )
    groups, library = _grouped_summaries(scenarios, manifest)
    grid = _grid_summary(scenarios)
    scenario_rows = (row.model_dump(mode="json") for row in scenarios)
    group_rows = _grid_csv_rows(groups)
    grid_rows = _grid_csv_rows(grid)
    missing = {metric: sum(getattr(row, metric) is None for row in scenarios) for metric in _METRICS}
    payload = {
        **library.model_dump(mode="json"),
        "run_id": descriptor.run_id,
        "framework": descriptor.framework,
        "model_id": descriptor.model_id,
        "session_id": descriptor.session_id,
        "checkpoint_id": descriptor.checkpoint_id,
        "checkpoint_sha256": descriptor.checkpoint_sha256,
        "library_manifest_sha256": descriptor.library_manifest_sha256,
        "run_configuration_sha256": descriptor.run_configuration_sha256,
        "deterministic_inference": descriptor.deterministic_inference,
        "repetitions_per_scenario": descriptor.repetitions_per_scenario,
        "sweep_status": descriptor.status.value,
        "publication_ready": True,
        "metric_coverage_complete": not any(missing.values()),
        "group_count": len(groups),
        "grid_cell_count": len(grid),
        "missing_metric_counts": missing,
        "invalid_trace_value_counts": dict(sorted(diagnostics.invalid_values.items())),
        "scenario_summary": SCENARIO_SUMMARY_FILENAME,
        "group_summary": GROUP_SUMMARY_FILENAME,
        "grid_summary": GRID_SUMMARY_FILENAME,
    }
    files = {
        SCENARIO_SUMMARY_FILENAME: _csv_bytes(
            tuple(GroupedScenarioSummary.model_fields), scenario_rows
        ),
        GROUP_SUMMARY_FILENAME: _csv_bytes(tuple(group_rows[0]), group_rows),
        GRID_SUMMARY_FILENAME: _csv_bytes(tuple(grid_rows[0]), grid_rows),
        GROUPED_LIBRARY_SUMMARY_FILENAME: (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    summary_root = publish_grouped_report(root, files)
    return EvaluationReportResult(summary_root, scenarios, grid, groups, library)


class EvaluationReportService:
    """Validate one terminal full sweep and write its narrow publication tables."""

    def generate(self, run_root: str | os.PathLike[str]) -> EvaluationReportResult:
        """Validate and atomically regenerate all report artifacts for one sweep."""

        root = Path(run_root).resolve()
        descriptor = _read_model(root / EVALUATION_LIBRARY_RUN_FILENAME, EvaluationSweepDescriptor)
        if descriptor.run_configuration_sha256 is not None:
            if descriptor.status is not EvaluationSweepStatus.COMPLETED:
                raise RuntimeError("Grouped publication requires a completed full-library sweep.")
            return _grouped_report(root, descriptor)
        if descriptor.status in {
            EvaluationSweepStatus.RUNNING,
            EvaluationSweepStatus.CANCELLED,
        } or (
            descriptor.completed_scenario_count + descriptor.failed_scenario_count
            != descriptor.expected_scenario_count
        ):
            raise RuntimeError("Evaluation report requires a terminal full-library sweep.")
        records = _read_scenario_records(root, descriptor)
        diagnostics = _Diagnostics()
        scenarios = tuple(_scenario_summary(root, record, diagnostics) for record in records)
        grid = _grid_summary(scenarios)
        summary_root = root / "summary"
        if os.path.lexists(summary_root):
            if summary_root.is_symlink() or not summary_root.is_dir():
                raise RuntimeError("Evaluation summary root is unsafe.")
        else:
            summary_root.mkdir()
        scenario_fields = tuple(ScenarioSummary.model_fields)
        _atomic_bytes(
            summary_root / SCENARIO_SUMMARY_FILENAME,
            _csv_bytes(
                scenario_fields,
                (scenario.model_dump(mode="json") for scenario in scenarios),
            ),
        )
        grid_rows = _grid_csv_rows(grid)
        grid_fields = tuple(grid_rows[0]) if grid_rows else ()
        _atomic_bytes(
            summary_root / GRID_SUMMARY_FILENAME,
            _csv_bytes(grid_fields, grid_rows),
        )
        missing = {
            metric: sum(getattr(row, metric) is None for row in scenarios) for metric in _METRICS
        }
        payload = {
            "schema_version": 1,
            "run_id": descriptor.run_id,
            "framework": descriptor.framework,
            "model_id": descriptor.model_id,
            "session_id": descriptor.session_id,
            "checkpoint_id": descriptor.checkpoint_id,
            "checkpoint_sha256": descriptor.checkpoint_sha256,
            "library_id": descriptor.library_id,
            "library_manifest_sha256": descriptor.library_manifest_sha256,
            "deterministic_inference": descriptor.deterministic_inference,
            "repetitions_per_scenario": descriptor.repetitions_per_scenario,
            "sweep_status": descriptor.status.value,
            "publication_ready": descriptor.status is EvaluationSweepStatus.COMPLETED,
            "metric_coverage_complete": not any(missing.values()),
            "expected_scenario_count": descriptor.expected_scenario_count,
            "attempted_scenario_count": len(scenarios),
            "completed_execution_count": descriptor.completed_scenario_count,
            "failed_execution_count": descriptor.failed_scenario_count,
            "completed_trip_count": sum(row.trip_completed is True for row in scenarios),
            "grid_cell_count": len(grid),
            "missing_metric_counts": missing,
            "invalid_trace_value_counts": dict(sorted(diagnostics.invalid_values.items())),
            "scenario_summary": SCENARIO_SUMMARY_FILENAME,
            "grid_summary": GRID_SUMMARY_FILENAME,
        }
        _atomic_bytes(
            summary_root / LIBRARY_SUMMARY_FILENAME,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return EvaluationReportResult(summary_root, scenarios, grid)


__all__ = [
    "EvaluationReportResult",
    "EvaluationReportService",
    "GROUP_SUMMARY_FILENAME",
    "GROUPED_LIBRARY_SUMMARY_FILENAME",
    "GRID_SUMMARY_FILENAME",
    "LIBRARY_SUMMARY_FILENAME",
    "SCENARIO_SUMMARY_FILENAME",
]
