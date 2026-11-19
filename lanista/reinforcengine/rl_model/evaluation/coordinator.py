# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate complete evaluation-library traversal above framework adapters."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from lanista.reinforcengine.rl_model.evaluation.contracts import (
    EvaluationGroupProgress,
    EvaluationScenarioExecutionRecord,
    EvaluationSweepDescriptor,
    EvaluationSweepProgress,
    EvaluationSweepStatus,
    RunConfigurationSnapshot,
    ScenarioExecutionStatus,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
    EvaluationLibraryManifest,
    EvaluationScenarioEntry,
    GroupedEvaluationLibraryManifest,
    canonical_manifest_bytes,
    validate_library_id,
    validate_manifest_payload,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.publication import (
    EVALUATION_LIBRARY_MANIFEST_FILENAME,
    read_verified_library,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.storage import (
    EvaluationLibraryStorageConfiguration,
)

EVALUATION_LIBRARY_RUN_FILENAME = "evaluation-library-run.json"
RUN_CONFIGURATION_FILENAME = "run-configuration.json"
LIBRARY_MANIFEST_SNAPSHOT_FILENAME = "library-manifest.json"
PARTIAL_SUMMARY_FILENAME = "partial-summary.json"
SCENARIO_RESULT_FILENAME = "scenario-result.json"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PRIMARY_EVIDENCE_FILENAMES = frozenset({"evaluation-run.json", "evaluation_trace.csv"})


class EvaluationSweepCancelled(RuntimeError):
    """Signal cancellation from inside one framework adapter."""


@dataclass(frozen=True, slots=True)
class EvaluationSweepRequest:
    """Freeze the checkpoint, library, inference, and output identity of a sweep."""

    run_id: str
    framework: Literal["ray", "sb3"]
    model_id: str
    session_id: str
    checkpoint_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    library_id: str
    library_manifest_sha256: str
    output_root: Path
    deterministic_inference: bool
    repetitions_per_scenario: int
    software_versions: Mapping[str, str]
    run_configuration: RunConfigurationSnapshot | None = None


@dataclass(frozen=True, slots=True)
class ScenarioEvaluationRequest:
    """Give a framework adapter one scenario and one private evidence scope."""

    scenario: EvaluationScenarioEntry
    scenario_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    evidence_root: Path
    deterministic_inference: bool
    repetitions: int
    cancel_requested: Callable[[], bool]


@dataclass(frozen=True, slots=True)
class EvaluationScenarioOutcome:
    """Return the terminal outcome after the adapter closes all evidence writers."""

    terminal_outcome: str


class ScenarioEvaluationAdapter(Protocol):
    """Evaluate exactly one explicitly selected scenario with a frozen policy."""

    def evaluate_scenario(
        self,
        request: ScenarioEvaluationRequest,
    ) -> EvaluationScenarioOutcome:
        """Run inference once and close every evidence writer before returning."""
        ...


@dataclass(frozen=True, slots=True)
class EvaluationSweepResult:
    """Return one terminal descriptor and its run-owned evidence root."""

    run_root: Path
    descriptor: EvaluationSweepDescriptor
    scenario_records: tuple[EvaluationScenarioExecutionRecord, ...]


@dataclass(frozen=True, slots=True)
class _ScenarioAttempt:
    """Carry one persisted attempt back to the traversal loop."""

    record: EvaluationScenarioExecutionRecord
    completed_increment: int = 0
    failed_increment: int = 0
    cancelled: bool = False
    failure_summary: str | None = None


@dataclass(slots=True)
class _SweepProgress:
    records: list[EvaluationScenarioExecutionRecord]
    completed_count: int = 0
    failed_count: int = 0
    cancelled: bool = False
    failure_summaries: list[str] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_digest(path: Path) -> str:
    """Return a stable SHA-256 identity for one file or directory checkpoint."""

    checkpoint = Path(path)
    if checkpoint.is_symlink():
        raise RuntimeError("Checkpoint paths must not be symbolic links.")
    if checkpoint.is_file():
        return _sha256_file(checkpoint)
    if not checkpoint.is_dir():
        raise RuntimeError(f"Checkpoint does not exist: {checkpoint!s}.")
    digest = hashlib.sha256()
    entries = tuple(checkpoint.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise RuntimeError("Checkpoint directories must not contain symbolic links.")
    if any(not item.is_file() and not item.is_dir() for item in entries):
        raise RuntimeError("Checkpoint directories may contain only regular files.")
    files = sorted(item for item in entries if item.is_file())
    if not files:
        raise RuntimeError("Directory checkpoint must contain regular files.")
    for item in files:
        relative = item.relative_to(checkpoint).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _canonical_model_bytes(model: BaseModel) -> bytes:
    payload = model.model_dump(mode="json", exclude_none=False)
    if isinstance(model, EvaluationSweepDescriptor) and model.run_configuration_sha256 is None:
        payload.pop("run_configuration_sha256", None)
    return _canonical_json_bytes(payload)


def _write_atomic(path: Path, model: BaseModel) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_model_bytes(model))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new(path: Path, model: BaseModel) -> None:
    _write_new_bytes(path, _canonical_model_bytes(model))


def _write_new_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_new_bytes(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _evidence_inventory(root: Path) -> tuple[str, ...]:
    evidence: list[str] = []
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for directory_name in directory_names:
            if (current / directory_name).is_symlink():
                raise RuntimeError("Scenario evidence must not contain symbolic links.")
        for file_name in file_names:
            path = current / file_name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("Scenario evidence may contain only regular files.")
            if path.name != SCENARIO_RESULT_FILENAME:
                evidence.append(path.relative_to(root).as_posix())
    return tuple(sorted(evidence))


def _require_primary_evidence(evidence_files: tuple[str, ...]) -> None:
    present = {Path(path).name for path in evidence_files}
    missing = sorted(_PRIMARY_EVIDENCE_FILENAMES - present)
    if missing:
        raise RuntimeError(f"Scenario evaluation evidence is incomplete; missing={missing}.")


def _failure_evidence_inventory(root: Path) -> tuple[str, ...]:
    """Retain readable evidence without allowing malformed artifacts to mask failure."""

    try:
        return _evidence_inventory(root)
    except (OSError, RuntimeError):
        return ()


def _validated_component(label: str, value: str) -> str:
    if _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"Unsafe {label}: {value!r}.")
    return value


def _validate_sweep_request(request: EvaluationSweepRequest) -> None:
    _validated_component("run_id", request.run_id)
    _validated_component("session_id", request.session_id)
    validate_library_id(request.library_id)
    if request.framework not in {"ray", "sb3"}:
        raise ValueError(f"Unsupported evaluation framework: {request.framework!r}.")
    if request.repetitions_per_scenario < 1:
        raise ValueError("repetitions_per_scenario must be positive.")
    if re.fullmatch(r"[0-9a-f]{64}", request.checkpoint_sha256) is None:
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest.")
    if checkpoint_digest(request.checkpoint_path) != request.checkpoint_sha256:
        raise RuntimeError("Selected checkpoint digest does not match the sweep request.")
    snapshot = request.run_configuration
    if snapshot is not None:
        expected = (
            (snapshot.model_id, request.model_id),
            (snapshot.framework, request.framework),
            (snapshot.session_id, request.session_id),
            (snapshot.checkpoint_id, request.checkpoint_id),
            (snapshot.checkpoint_sha256, request.checkpoint_sha256),
            (snapshot.library_id, request.library_id),
            (snapshot.library_manifest_sha256, request.library_manifest_sha256),
            (snapshot.deterministic_inference, request.deterministic_inference),
            (snapshot.repetitions_per_scenario, request.repetitions_per_scenario),
            (snapshot.software_versions, dict(request.software_versions)),
        )
        if any(actual != selected for actual, selected in expected):
            raise ValueError("Run configuration snapshot must match the selected sweep.")


def _terminal_status(
    *,
    cancelled: bool,
    checkpoint_changed: bool,
    failure_summaries: list[str],
) -> EvaluationSweepStatus:
    """Resolve integrity failure, cancellation, scenario failure, then success."""

    if checkpoint_changed:
        return EvaluationSweepStatus.FAILED
    if cancelled:
        return EvaluationSweepStatus.CANCELLED
    if failure_summaries:
        return EvaluationSweepStatus.FAILED
    return EvaluationSweepStatus.COMPLETED


def _descriptor(
    request: EvaluationSweepRequest,
    *,
    started_at: str,
    ended_at: str | None,
    status: EvaluationSweepStatus,
    expected_count: int,
    completed_count: int,
    failed_count: int,
    failure_message: str | None = None,
) -> EvaluationSweepDescriptor:
    return EvaluationSweepDescriptor(
        schema_version=1,
        run_id=request.run_id,
        framework=request.framework,
        model_id=request.model_id,
        session_id=request.session_id,
        checkpoint_id=request.checkpoint_id,
        checkpoint_sha256=request.checkpoint_sha256,
        library_id=request.library_id,
        library_manifest_sha256=request.library_manifest_sha256,
        run_configuration_sha256=(
            hashlib.sha256(_canonical_model_bytes(request.run_configuration)).hexdigest()
            if request.run_configuration is not None
            else None
        ),
        deterministic_inference=request.deterministic_inference,
        repetitions_per_scenario=request.repetitions_per_scenario,
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        expected_scenario_count=expected_count,
        completed_scenario_count=completed_count,
        failed_scenario_count=failed_count,
        software_versions=dict(sorted(request.software_versions.items())),
        failure_message=failure_message,
    )


def _checked_run_parent(path: Path) -> Path:
    """Create a run parent without following a pre-existing symlink alias."""

    parent = Path(path).absolute()
    if any(component.is_symlink() for component in (parent, *parent.parents)):
        raise RuntimeError("Evaluation run parent contains a symlink.")
    parent.mkdir(parents=True, exist_ok=True)
    if any(component.is_symlink() for component in (parent, *parent.parents)):
        raise RuntimeError("Evaluation run parent contains a symlink.")
    return parent


def _persisted_progress(
    run_root: Path,
    manifest: GroupedEvaluationLibraryManifest,
    run_id: str,
) -> EvaluationSweepProgress:
    """Derive counts only from already closed scenario-result files."""

    records = tuple(
        EvaluationScenarioExecutionRecord.model_validate_json(path.read_bytes())
        for path in sorted((run_root / "scenarios").glob(f"*/{SCENARIO_RESULT_FILENAME}"))
    )
    by_group = {entry.scenario_id: entry.group_id for entry in manifest.scenarios}
    if any(record.scenario_id not in by_group for record in records):
        raise RuntimeError("Persisted scenario record is absent from the grouped manifest.")
    groups = []
    for group in manifest.groups:
        members = tuple(record for record in records if by_group[record.scenario_id] == group.group_id)
        completed = sum(record.status is ScenarioExecutionStatus.COMPLETED for record in members)
        failed = sum(record.status is ScenarioExecutionStatus.FAILED for record in members)
        cancelled = sum(record.status is ScenarioExecutionStatus.CANCELLED for record in members)
        groups.append(
            EvaluationGroupProgress(
                group_id=group.group_id,
                name=group.name,
                expected_scenario_count=group.scenario_count,
                completed_scenario_count=completed,
                failed_scenario_count=failed,
                cancelled_scenario_count=cancelled,
                remaining_scenario_count=group.scenario_count - completed - failed - cancelled,
            )
        )
    completed = sum(group.completed_scenario_count for group in groups)
    failed = sum(group.failed_scenario_count for group in groups)
    cancelled = sum(group.cancelled_scenario_count for group in groups)
    return EvaluationSweepProgress(
        run_id=run_id,
        expected_scenario_count=manifest.expected_scenario_count,
        completed_scenario_count=completed,
        failed_scenario_count=failed,
        cancelled_scenario_count=cancelled,
        remaining_scenario_count=manifest.expected_scenario_count - completed - failed - cancelled,
        groups=tuple(groups),
    )


def _write_partial_summary(
    run_root: Path,
    descriptor: EvaluationSweepDescriptor,
    manifest: GroupedEvaluationLibraryManifest,
) -> None:
    """Keep failed/cancelled grouped counts separate from publication reports."""

    summary_root = run_root / "summary"
    summary_root.mkdir(exist_ok=True)
    if summary_root.is_symlink() or any(
        path.name != PARTIAL_SUMMARY_FILENAME for path in summary_root.iterdir()
    ):
        raise RuntimeError("Partial evaluation summary root is unsafe.")
    progress = _persisted_progress(run_root, manifest, descriptor.run_id)
    payload = {
        **progress.model_dump(mode="json"),
        "schema_version": 1,
        "status": descriptor.status.value,
        "publication_ready": False,
        "failure_message": descriptor.failure_message,
    }
    _write_atomic_bytes(
        summary_root / PARTIAL_SUMMARY_FILENAME,
        _canonical_json_bytes(payload),
    )


class EvaluationSweepCoordinator:
    """Traverse one verified library while isolating framework-owned evidence."""

    def __init__(
        self,
        library_storage: EvaluationLibraryStorageConfiguration,
        *,
        utc_now: Callable[[], str],
    ) -> None:
        self._library_storage = library_storage
        self._utc_now = utc_now

    def run(
        self,
        request: EvaluationSweepRequest,
        adapter: ScenarioEvaluationAdapter,
        *,
        cancel_requested: Callable[[], bool] = lambda: False,
        progress_callback: Callable[[EvaluationSweepProgress], None] | None = None,
    ) -> EvaluationSweepResult:
        """Validate inputs, visit manifest order once, and persist terminal state."""

        _validate_sweep_request(request)
        manifest = read_verified_library(
            self._library_storage,
            request.library_id,
            expected_manifest_sha256=request.library_manifest_sha256,
        )
        if isinstance(manifest, GroupedEvaluationLibraryManifest) != (
            request.run_configuration is not None
        ):
            raise ValueError("Grouped runs require a frozen run configuration.")
        run_root, scenarios_root, descriptor_path, started_at = self._initialize_run(
            request,
            manifest=manifest,
        )
        progress = self._traverse_scenarios(
            request=request,
            adapter=adapter,
            cancel_requested=cancel_requested,
            scenarios=manifest.scenarios,
            scenarios_root=scenarios_root,
            descriptor_path=descriptor_path,
            started_at=started_at,
            grouped_manifest=(
                manifest if isinstance(manifest, GroupedEvaluationLibraryManifest) else None
            ),
            progress_callback=progress_callback,
        )
        return self._finish_sweep(
            request=request,
            run_root=run_root,
            descriptor_path=descriptor_path,
            started_at=started_at,
            expected_count=len(manifest.scenarios),
            progress=progress,
            grouped_manifest=(
                manifest if isinstance(manifest, GroupedEvaluationLibraryManifest) else None
            ),
        )

    def _initialize_run(
        self,
        request: EvaluationSweepRequest,
        *,
        manifest: EvaluationLibraryManifest | GroupedEvaluationLibraryManifest,
    ) -> tuple[Path, Path, Path, str]:
        """Create the run-owned roots and initial running descriptor."""

        parent = (
            Path(request.output_root)
            if request.run_configuration is not None
            else Path(request.output_root) / request.framework / request.session_id
        )
        run_root = _checked_run_parent(parent) / request.run_id
        run_root.mkdir()
        scenarios_root = run_root / "scenarios"
        scenarios_root.mkdir()
        started_at = self._utc_now()
        descriptor_path = run_root / EVALUATION_LIBRARY_RUN_FILENAME
        running = _descriptor(
            request,
            started_at=started_at,
            ended_at=None,
            status=EvaluationSweepStatus.RUNNING,
            expected_count=len(manifest.scenarios),
            completed_count=0,
            failed_count=0,
        )
        _write_atomic(descriptor_path, running)
        if request.run_configuration is not None:
            _write_new_bytes(
                run_root / RUN_CONFIGURATION_FILENAME,
                _canonical_model_bytes(request.run_configuration),
            )
            _write_new_bytes(
                run_root / LIBRARY_MANIFEST_SNAPSHOT_FILENAME,
                canonical_manifest_bytes(manifest),
            )
        return run_root, scenarios_root, descriptor_path, started_at

    def _traverse_scenarios(
        self,
        *,
        request: EvaluationSweepRequest,
        adapter: ScenarioEvaluationAdapter,
        cancel_requested: Callable[[], bool],
        scenarios: tuple[EvaluationScenarioEntry, ...],
        scenarios_root: Path,
        descriptor_path: Path,
        started_at: str,
        grouped_manifest: GroupedEvaluationLibraryManifest | None,
        progress_callback: Callable[[EvaluationSweepProgress], None] | None,
    ) -> _SweepProgress:
        """Visit manifest order once and persist progress after every attempt."""

        progress = _SweepProgress(records=[])
        library_root = self._library_storage.library_path(request.library_id)
        for sequence_index, scenario in enumerate(scenarios):
            if cancel_requested():
                progress.cancelled = True
                break
            attempt = self._execute_scenario(
                sequence_index=sequence_index,
                scenario=scenario,
                scenarios_root=scenarios_root,
                library_root=library_root,
                request=request,
                adapter=adapter,
                cancel_requested=cancel_requested,
            )
            progress.completed_count += attempt.completed_increment
            progress.failed_count += attempt.failed_increment
            progress.cancelled = attempt.cancelled
            if attempt.failure_summary is not None:
                progress.failure_summaries.append(attempt.failure_summary)
            progress.records.append(attempt.record)
            running = _descriptor(
                request,
                started_at=started_at,
                ended_at=None,
                status=EvaluationSweepStatus.RUNNING,
                expected_count=len(scenarios),
                completed_count=progress.completed_count,
                failed_count=progress.failed_count,
            )
            _write_atomic(descriptor_path, running)
            if grouped_manifest is not None and progress_callback is not None:
                try:
                    progress_callback(
                        _persisted_progress(
                            scenarios_root.parent, grouped_manifest, request.run_id
                        )
                    )
                except Exception:
                    logging.exception("Evaluation progress observer failed.")
            if progress.cancelled:
                break
        return progress

    def _finish_sweep(
        self,
        *,
        request: EvaluationSweepRequest,
        run_root: Path,
        descriptor_path: Path,
        started_at: str,
        expected_count: int,
        progress: _SweepProgress,
        grouped_manifest: GroupedEvaluationLibraryManifest | None,
    ) -> EvaluationSweepResult:
        """Validate persisted attempts and publish one terminal descriptor."""

        self._validate_persisted_records(run_root, progress.records)
        checkpoint_changed = checkpoint_digest(request.checkpoint_path) != request.checkpoint_sha256
        if checkpoint_changed:
            progress.failure_summaries.append("Checkpoint content changed during the sweep.")
        terminal_status = _terminal_status(
            cancelled=progress.cancelled,
            checkpoint_changed=checkpoint_changed,
            failure_summaries=progress.failure_summaries,
        )
        terminal = _descriptor(
            request,
            started_at=started_at,
            ended_at=self._utc_now(),
            status=terminal_status,
            expected_count=expected_count,
            completed_count=progress.completed_count,
            failed_count=progress.failed_count,
            failure_message=(
                "; ".join(progress.failure_summaries)
                if terminal_status is EvaluationSweepStatus.FAILED
                else None
            ),
        )
        _write_atomic(descriptor_path, terminal)
        if grouped_manifest is not None and terminal.status is not EvaluationSweepStatus.COMPLETED:
            _write_partial_summary(run_root, terminal, grouped_manifest)
        return EvaluationSweepResult(
            run_root=run_root,
            descriptor=terminal,
            scenario_records=tuple(progress.records),
        )

    def _execute_scenario(
        self,
        *,
        sequence_index: int,
        scenario: EvaluationScenarioEntry,
        scenarios_root: Path,
        library_root: Path,
        request: EvaluationSweepRequest,
        adapter: ScenarioEvaluationAdapter,
        cancel_requested: Callable[[], bool],
    ) -> _ScenarioAttempt:
        """Execute and persist one scenario without leaking errors to later entries."""

        evidence_root = scenarios_root / scenario.scenario_id
        evidence_root.mkdir()
        scenario_started_at = self._utc_now()
        try:
            outcome = adapter.evaluate_scenario(
                ScenarioEvaluationRequest(
                    scenario=scenario,
                    scenario_root=library_root / scenario.relative_path,
                    checkpoint_path=request.checkpoint_path,
                    checkpoint_sha256=request.checkpoint_sha256,
                    evidence_root=evidence_root,
                    deterministic_inference=request.deterministic_inference,
                    repetitions=request.repetitions_per_scenario,
                    cancel_requested=cancel_requested,
                )
            )
            evidence_files = _evidence_inventory(evidence_root)
            _require_primary_evidence(evidence_files)
            attempt = _ScenarioAttempt(
                record=self._scenario_record(
                    sequence_index,
                    scenario,
                    status=ScenarioExecutionStatus.COMPLETED,
                    started_at=scenario_started_at,
                    evidence_files=evidence_files,
                    terminal_outcome=outcome.terminal_outcome,
                ),
                completed_increment=1,
            )
        except EvaluationSweepCancelled:
            attempt = _ScenarioAttempt(
                record=self._scenario_record(
                    sequence_index,
                    scenario,
                    status=ScenarioExecutionStatus.CANCELLED,
                    started_at=scenario_started_at,
                    evidence_files=_failure_evidence_inventory(evidence_root),
                ),
                cancelled=True,
            )
        except Exception as exc:
            failure_message = str(exc) or repr(exc)
            attempt = _ScenarioAttempt(
                record=self._scenario_record(
                    sequence_index,
                    scenario,
                    status=ScenarioExecutionStatus.FAILED,
                    started_at=scenario_started_at,
                    evidence_files=_failure_evidence_inventory(evidence_root),
                    failure_type=type(exc).__name__,
                    failure_message=failure_message,
                ),
                failed_increment=1,
                failure_summary=(
                    f"{scenario.scenario_id}: {type(exc).__name__}: {failure_message}"
                ),
            )
        _write_new(evidence_root / SCENARIO_RESULT_FILENAME, attempt.record)
        return attempt

    def _scenario_record(
        self,
        sequence_index: int,
        scenario: EvaluationScenarioEntry,
        *,
        status: ScenarioExecutionStatus,
        started_at: str,
        evidence_files: tuple[str, ...],
        terminal_outcome: str | None = None,
        failure_type: str | None = None,
        failure_message: str | None = None,
    ) -> EvaluationScenarioExecutionRecord:
        return EvaluationScenarioExecutionRecord(
            schema_version=1,
            sequence_index=sequence_index,
            scenario_id=scenario.scenario_id,
            requested_vehicle_count=scenario.requested_vehicle_count,
            requested_pedestrian_count=scenario.requested_pedestrian_count,
            replicate_index=scenario.replicate_index,
            seed=scenario.seed,
            status=status,
            started_at=started_at,
            ended_at=self._utc_now(),
            evidence_files=evidence_files,
            terminal_outcome=terminal_outcome,
            failure_type=failure_type,
            failure_message=failure_message,
        )

    @staticmethod
    def _validate_persisted_records(
        run_root: Path,
        records: list[EvaluationScenarioExecutionRecord],
    ) -> None:
        for expected in records:
            path = run_root / "scenarios" / expected.scenario_id / SCENARIO_RESULT_FILENAME
            restored = EvaluationScenarioExecutionRecord.model_validate_json(path.read_bytes())
            if restored != expected:
                raise RuntimeError(f"Scenario result readback changed: {expected.scenario_id}.")


def fail_report_publication(
    result: EvaluationSweepResult,
    message: str,
) -> EvaluationSweepResult:
    """Persist a grouped exporter failure without discarding scenario evidence."""

    descriptor = result.descriptor
    if descriptor.run_configuration_sha256 is None or descriptor.status is not EvaluationSweepStatus.COMPLETED:
        raise ValueError("Only a completed grouped run can fail report publication.")
    failure = str(message).strip() or "Unknown report publication failure."
    failed = EvaluationSweepDescriptor.model_validate(
        {
            **descriptor.model_dump(mode="json"),
            "status": EvaluationSweepStatus.FAILED,
            "failure_message": f"Report publication failed: {failure}",
        }
    )
    _write_atomic(result.run_root / EVALUATION_LIBRARY_RUN_FILENAME, failed)
    try:
        manifest_bytes = (result.run_root / LIBRARY_MANIFEST_SNAPSHOT_FILENAME).read_bytes()
        manifest = validate_manifest_payload(json.loads(manifest_bytes))
        if (
            not isinstance(manifest, GroupedEvaluationLibraryManifest)
            or canonical_manifest_bytes(manifest) != manifest_bytes
            or hashlib.sha256(manifest_bytes).hexdigest() != descriptor.library_manifest_sha256
        ):
            raise RuntimeError("Frozen grouped manifest is unavailable for partial reporting.")
        _write_partial_summary(result.run_root, failed, manifest)
    except Exception:
        logging.exception("Could not publish per-group partial summary after report failure.")
        summary_root = result.run_root / "summary"
        summary_root.mkdir(exist_ok=True)
        attempted = len(result.scenario_records)
        payload = {
            "schema_version": 1,
            "run_id": descriptor.run_id,
            "status": failed.status.value,
            "publication_ready": False,
            "expected_scenario_count": descriptor.expected_scenario_count,
            "attempted_scenario_count": attempted,
            "completed_scenario_count": descriptor.completed_scenario_count,
            "failed_scenario_count": descriptor.failed_scenario_count,
            "remaining_scenario_count": descriptor.expected_scenario_count - attempted,
            "groups": [],
            "failure_message": failed.failure_message,
        }
        _write_atomic_bytes(summary_root / PARTIAL_SUMMARY_FILENAME, _canonical_json_bytes(payload))
    return EvaluationSweepResult(result.run_root, failed, result.scenario_records)


__all__ = [
    "EVALUATION_LIBRARY_RUN_FILENAME",
    "SCENARIO_RESULT_FILENAME",
    "EvaluationScenarioOutcome",
    "EvaluationSweepCancelled",
    "EvaluationSweepCoordinator",
    "EvaluationSweepRequest",
    "EvaluationSweepResult",
    "ScenarioEvaluationAdapter",
    "ScenarioEvaluationRequest",
    "checkpoint_digest",
    "fail_report_publication",
]
