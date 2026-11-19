# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Bind one explicit library scope to a managed runtime execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lanista.reinforcengine.rl_model.evaluation.contracts import (
    EvaluationSweepProgress,
    EvaluationSweepStatus,
    LibraryEvaluationScope,
)
from lanista.reinforcengine.rl_model.evaluation.coordinator import (
    EvaluationSweepRequest,
    EvaluationSweepResult,
)
from lanista.reinforcengine.rl_model.metadata.timestamps import utc_now_iso
from lanista.reinforcengine.utils.scenarios.evaluation_library.storage import (
    EvaluationLibraryStorageConfiguration,
)


@dataclass(frozen=True, slots=True)
class LibraryEvaluationExecution:
    """Carry one internally consistent library selection and sweep request."""

    scope: LibraryEvaluationScope
    library_storage: EvaluationLibraryStorageConfiguration
    sweep_request: EvaluationSweepRequest
    utc_now: Callable[[], str] = utc_now_iso
    progress_callback: Callable[[EvaluationSweepProgress], None] | None = None
    report_publisher: Callable[[Path], object] | None = None

    def __post_init__(self) -> None:
        request = self.sweep_request
        mismatches = (
            self.scope.library_id != request.library_id,
            self.scope.manifest_sha256 != request.library_manifest_sha256,
            self.scope.repetitions_per_scenario != request.repetitions_per_scenario,
        )
        if any(mismatches):
            raise ValueError(
                "Library evaluation scope must match the sweep library, digest, "
                "and repetition count."
            )
        if request.run_configuration is not None and self.report_publisher is None:
            raise ValueError("Grouped evaluation requires an injected report publisher.")


class EvaluationLibrarySweepFailed(RuntimeError):
    """Keep a failed sweep's persisted result attached to runtime failure."""

    def __init__(self, result: EvaluationSweepResult) -> None:
        self.result = result
        message = result.descriptor.failure_message or "Evaluation-library sweep failed."
        super().__init__(message)


def require_successful_sweep(result: EvaluationSweepResult) -> EvaluationSweepResult:
    """Raise after evidence persistence when a sweep terminally failed."""

    if result.descriptor.status is EvaluationSweepStatus.FAILED:
        raise EvaluationLibrarySweepFailed(result)
    return result


__all__ = [
    "EvaluationLibrarySweepFailed",
    "LibraryEvaluationExecution",
    "require_successful_sweep",
]
