# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Best-effort CSV and raw TensorBoard persistence for evaluation traces."""

from __future__ import annotations

import csv
import json
import math
import re
import uuid
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, cast

from lanista.reinforcengine.rl_metrics.diagnostic_trace_contracts import (
    DiagnosticTraceRow,
    DiagnosticTraceSource,
    DiagnosticTraceTensorboardWriter,
    QualityReporter,
)
from lanista.reinforcengine.rl_metrics.tensorboard_schema import (
    evaluation_tensorboard_tag,
)

_TRACE_DIRECTORY_NAME = "evaluation_traces"
_TRACE_FILE_NAME = "evaluation_trace.csv"
_EVALUATION_DESCRIPTOR_FILE_NAME = "evaluation-run.json"
_BASE_FIELDS = (
    "trace_run_id",
    "framework",
    "mode",
    "episode_index",
    "terminated",
    "truncated",
)
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_OPTIONAL_TRACE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _open_trace_file(path: str) -> TextIO:
    """Open a new UTF-8 trace file without replacing prior evidence."""

    return open(path, "x", encoding="utf-8", newline="")


def _runtime_call(runtime: object, name: str) -> object | None:
    """Call one optional zero-argument runtime projection."""

    provider = getattr(runtime, name, None)
    return provider() if callable(provider) else None


def _write_evaluation_descriptor(
    *,
    log_dir: str,
    runtime: object,
    framework: str,
) -> None:
    """Create the immutable human-readable descriptor for one evaluation."""

    manifest = _runtime_call(runtime, "get_manifest")
    configuration_provider = getattr(manifest, "get_configuration", None)
    configuration = (
        configuration_provider()
        if callable(configuration_provider)
        else None
    )
    descriptor_path = Path(log_dir).resolve() / _EVALUATION_DESCRIPTOR_FILE_NAME
    if descriptor_path.exists():
        return
    scenario_index = _runtime_call(runtime, "get_evaluation_scenario_index")
    visualization = _runtime_call(runtime, "get_evaluation_visualization")
    repetitions = _runtime_call(runtime, "get_evaluation_repetition_count")
    descriptor = {
        "schema_version": 1,
        "evaluation_id": Path(log_dir).name,
        "created_at": datetime.now(UTC).isoformat(),
        "framework": str(framework),
        "session": str(
            getattr(configuration, "selected_default_model_name", "") or ""
        ),
        "checkpoint": str(
            getattr(configuration, "selected_default_checkpoint_name", "") or ""
        ),
        "scenario_index": (
            int(cast(float | int | str, scenario_index))
            if scenario_index is not None
            else None
        ),
        "visualization": str(visualization) if visualization is not None else None,
        "repetitions": (
            int(cast(float | int | str, repetitions))
            if repetitions is not None
            else None
        ),
    }
    with descriptor_path.open("x", encoding="utf-8") as descriptor_file:
        json.dump(descriptor, descriptor_file, indent=2, sort_keys=True)
        descriptor_file.write("\n")


class EvaluationTraceRun:
    """Own one optional raw evaluation trace and its TensorBoard projection."""

    def __init__(
        self,
        *,
        trace_run_id: str,
        framework: str,
        csv_path: str,
        fieldnames: tuple[str, ...],
        handle: TextIO | None,
        writer: csv.DictWriter | None,
        tensorboard_writer: DiagnosticTraceTensorboardWriter | None,
        tensorboard_fields: frozenset[str],
        quality_reporter: QualityReporter | None,
    ) -> None:
        """Initialize one created or safely disabled evaluation trace."""

        self.trace_run_id = trace_run_id
        self.framework = framework
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self._handle = handle
        self._writer = writer
        self._tensorboard_writer = tensorboard_writer
        self._tensorboard_fields = tensorboard_fields
        self._quality_reporter = quality_reporter
        self._quality_degradations: set[str] = set()
        self._tensorboard_step = 0
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        log_dir: str,
        framework: str,
        fieldnames: Iterable[str],
        tensorboard_writer: DiagnosticTraceTensorboardWriter | None = None,
        tensorboard_fields: frozenset[str] = frozenset(),
        quality_reporter: QualityReporter | None = None,
        run_id_factory: Callable[[], str] | None = None,
    ) -> EvaluationTraceRun:
        """Create a unique evaluation trace or return a disabled trace."""

        trace_run_id = str((run_id_factory or (lambda: uuid.uuid4().hex))())
        requested_fields = tuple(dict.fromkeys(str(name) for name in fieldnames))
        csv_fields = tuple(dict.fromkeys((*_BASE_FIELDS, *requested_fields)))
        trace_root = Path(log_dir).resolve() / _TRACE_DIRECTORY_NAME
        csv_path = trace_root / trace_run_id / _TRACE_FILE_NAME
        trace: EvaluationTraceRun
        try:
            if not _SAFE_RUN_ID.fullmatch(trace_run_id):
                raise ValueError("Diagnostic trace run identifier is unsafe.")
            csv_path.parent.mkdir(parents=True, exist_ok=False)
            handle = _open_trace_file(str(csv_path))
            writer = csv.DictWriter(
                handle,
                fieldnames=csv_fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            trace = cls(
                trace_run_id=trace_run_id,
                framework=str(framework),
                csv_path=str(csv_path),
                fieldnames=csv_fields,
                handle=handle,
                writer=writer,
                tensorboard_writer=tensorboard_writer,
                tensorboard_fields=tensorboard_fields,
                quality_reporter=quality_reporter,
            )
        except _OPTIONAL_TRACE_ERRORS as error:
            trace = cls(
                trace_run_id=trace_run_id,
                framework=str(framework),
                csv_path=str(csv_path),
                fieldnames=csv_fields,
                handle=None,
                writer=None,
                tensorboard_writer=tensorboard_writer,
                tensorboard_fields=tensorboard_fields,
                quality_reporter=quality_reporter,
            )
            trace._report("evaluation_trace_create_failed", error)
        return trace

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return the stable set of optional trace failures."""

        return frozenset(self._quality_degradations)

    def append(
        self,
        row: DiagnosticTraceRow,
        *,
        episode_index: int,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        """Append one raw step and publish its selected finite scalars."""

        if self._closed:
            return
        output_row: dict[str, object] = {
            "trace_run_id": self.trace_run_id,
            "framework": self.framework,
            "mode": "evaluation",
            "episode_index": int(episode_index),
            "terminated": int(bool(terminated)),
            "truncated": int(bool(truncated)),
        }
        output_row.update(row)
        if self._writer is not None:
            try:
                self._writer.writerow(output_row)
            except _OPTIONAL_TRACE_ERRORS as error:
                self._report("evaluation_trace_write_failed", error)
        self._publish_tensorboard(
            row,
            episode_index=episode_index,
            terminated=terminated,
            truncated=truncated,
        )

    def _publish_tensorboard(
        self,
        row: DiagnosticTraceRow,
        *,
        episode_index: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Publish a curated projection on one execution-global step axis."""

        writer = self._tensorboard_writer
        if writer is None:
            return
        step = self._tensorboard_step
        self._tensorboard_step += 1
        for tag, numeric in (
            ("lanistai_eval_overview/episode_index", float(episode_index)),
            ("lanistai_eval_overview/terminated", float(bool(terminated))),
            ("lanistai_eval_overview/truncated", float(bool(truncated))),
        ):
            self._write_tensorboard_scalar(writer, tag, numeric, step)
        for field in self.fieldnames:
            if field not in self._tensorboard_fields:
                continue
            tag_name = evaluation_tensorboard_tag(field)
            if tag_name is None:
                continue
            value = row.get(field)
            try:
                numeric = float(cast(float | int | str, value))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            self._write_tensorboard_scalar(writer, tag_name, numeric, step)

    def _write_tensorboard_scalar(
        self,
        writer: DiagnosticTraceTensorboardWriter,
        tag: str,
        value: float,
        step: int,
    ) -> None:
        """Write one optional evaluation scalar through the common error path."""

        try:
            writer.add_scalar(tag, value, step)
        except _OPTIONAL_TRACE_ERRORS as error:
            self._report(
                "evaluation_trace_tensorboard_write_failed",
                error,
            )

    def flush(self) -> None:
        """Flush optional trace data without changing execution outcome."""

        if self._closed or self._handle is None:
            return
        try:
            self._handle.flush()
        except _OPTIONAL_TRACE_ERRORS as error:
            self._report("evaluation_trace_flush_failed", error)

    def close(self) -> None:
        """Flush and close once while containing optional persistence errors."""

        if self._closed:
            return
        self.flush()
        self._closed = True
        if self._handle is None:
            return
        try:
            self._handle.close()
        except _OPTIONAL_TRACE_ERRORS as error:
            self._report("evaluation_trace_close_failed", error)

    def _report(self, event: str, error: BaseException) -> None:
        """Report only the first occurrence of each quality degradation."""

        if event in self._quality_degradations:
            return
        self._quality_degradations.add(event)
        if self._quality_reporter is not None:
            with suppress(Exception):
                self._quality_reporter(event, error)


def _diagnostic_trace_providers(
    runtime: object,
    source: object,
) -> tuple[
    Callable[[], Iterable[str]],
    Callable[[], Iterable[str]],
    Callable[[], object],
    Callable[[], DiagnosticTraceTensorboardWriter | None],
] | None:
    """Resolve the complete optional diagnostic source and runtime contract."""

    source_type = type(source)
    fieldnames_provider = getattr(source_type, "diagnostic_trace_fieldnames", None)
    tensorboard_fields_provider = getattr(
        source_type,
        "diagnostic_trace_tensorboard_fields",
        None,
    )
    row_provider = getattr(source_type, "latest_diagnostic_trace_row", None)
    log_dir_provider = getattr(runtime, "get_resolved_tensorboard_log_dir", None)
    writer_provider = getattr(runtime, "get_tensorboard_writer", None)
    if not all(
        callable(provider)
        for provider in (
            fieldnames_provider,
            tensorboard_fields_provider,
            row_provider,
            log_dir_provider,
            writer_provider,
        )
    ):
        return None
    return (
        cast(Callable[[], Iterable[str]], fieldnames_provider),
        cast(Callable[[], Iterable[str]], tensorboard_fields_provider),
        cast(Callable[[], object], log_dir_provider),
        cast(
            Callable[[], DiagnosticTraceTensorboardWriter | None],
            writer_provider,
        ),
    )


class EvaluationTraceSession:
    """Attach one optional trace run to a framework-neutral evaluation loop."""

    def __init__(
        self,
        *,
        source: object | None = None,
        trace: EvaluationTraceRun | None = None,
        scenario_index_provider: Callable[[], int] | None = None,
        quality_reporter: QualityReporter | None = None,
    ) -> None:
        """Initialize an active session or an intentionally inert adapter."""

        self._source = source
        self._trace = trace
        self._scenario_index_provider = scenario_index_provider
        self._quality_reporter = quality_reporter
        self._quality_degradations: set[str] = set()

    @classmethod
    def create(
        cls,
        *,
        runtime: object,
        source: object,
        framework: str,
        quality_reporter: QualityReporter | None = None,
        run_id_factory: Callable[[], str] | None = None,
    ) -> EvaluationTraceSession:
        """Attach to a diagnostic source without requiring every model to expose one."""

        runtime_reporter = getattr(
            runtime,
            "report_diagnostic_quality_degradation",
            None,
        )
        resolved_reporter = quality_reporter or (
            runtime_reporter if callable(runtime_reporter) else None
        )
        providers = _diagnostic_trace_providers(runtime, source)
        if providers is None:
            return cls(quality_reporter=resolved_reporter)
        (
            fieldnames_provider,
            tensorboard_fields_provider,
            log_dir_provider,
            writer_provider,
        ) = providers

        try:
            log_dir = log_dir_provider()
            if not log_dir:
                return cls(quality_reporter=resolved_reporter)
            try:
                _write_evaluation_descriptor(
                    log_dir=str(log_dir),
                    runtime=runtime,
                    framework=framework,
                )
            except _OPTIONAL_TRACE_ERRORS as error:
                if resolved_reporter is not None:
                    with suppress(Exception):
                        resolved_reporter(
                            "evaluation_descriptor_write_failed",
                            error,
                        )
            trace = EvaluationTraceRun.create(
                log_dir=str(log_dir),
                framework=framework,
                fieldnames=fieldnames_provider(),
                tensorboard_writer=writer_provider(),
                tensorboard_fields=frozenset(tensorboard_fields_provider()),
                quality_reporter=resolved_reporter,
                run_id_factory=run_id_factory,
            )
        except Exception as error:
            session = cls(quality_reporter=resolved_reporter)
            session._report("evaluation_trace_attach_failed", error)
            return session

        scenario_index_provider = getattr(
            runtime,
            "get_evaluation_scenario_index",
            None,
        )
        return cls(
            source=source,
            trace=trace,
            scenario_index_provider=(
                scenario_index_provider
                if callable(scenario_index_provider)
                else None
            ),
            quality_reporter=resolved_reporter,
        )

    @property
    def active(self) -> bool:
        """Return whether the source and trace lifecycle were attached."""

        return self._source is not None and self._trace is not None

    @property
    def csv_path(self) -> str:
        """Return the trace path or an empty string for an inert session."""

        return self._trace.csv_path if self._trace is not None else ""

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return local and recorder-owned optional failure codes."""

        recorder_events = (
            self._trace.quality_degradations
            if self._trace is not None
            else frozenset()
        )
        return frozenset((*self._quality_degradations, *recorder_events))

    def append_latest(
        self,
        *,
        episode_index: int,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        """Project and append the latest completed source step when available."""

        if not self.active:
            return
        assert self._source is not None
        assert self._trace is not None
        row_provider = cast(
            DiagnosticTraceSource,
            self._source,
        ).latest_diagnostic_trace_row
        try:
            scenario_index = (
                self._scenario_index_provider()
                if self._scenario_index_provider is not None
                else None
            )
            row = row_provider(
                scenario_index=scenario_index,
                terminated=terminated,
                truncated=truncated,
            )
            if row is None:
                raise RuntimeError("No completed diagnostic step is available.")
            self._trace.append(
                row,
                episode_index=episode_index,
                terminated=terminated,
                truncated=truncated,
            )
        except Exception as error:
            self._report("evaluation_trace_projection_failed", error)

    def flush(self) -> None:
        """Flush the attached trace at an episode boundary."""

        if self._trace is not None:
            self._trace.flush()

    def close(self) -> None:
        """Close the attached trace idempotently."""

        if self._trace is not None:
            self._trace.close()

    def _report(self, event: str, error: BaseException) -> None:
        """Contain and report the first occurrence of one session failure."""

        if event in self._quality_degradations:
            return
        self._quality_degradations.add(event)
        if self._quality_reporter is not None:
            with suppress(Exception):
                self._quality_reporter(event, error)


__all__ = ["EvaluationTraceRun", "EvaluationTraceSession"]
