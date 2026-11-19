# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Bind one inference execution to its scenario-owned evidence directory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from lanista.reinforcengine.rl_model.evaluation.coordinator import (
    EvaluationSweepCancelled,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.runtime.tensorboard_writer import (
    SummaryWriterFactory,
    TensorboardWriter,
    build_tensorboard_writer,
)


class EvaluationRuntimeDelegate(Protocol):
    """Expose lifecycle and manifest operations retained from the active runtime."""

    def get_manifest(self) -> ModelManifest:
        """Return the selected model manifest."""
        ...

    def check_pause(self) -> None:
        """Honor application-level pause state."""
        ...

    def check_stop(self) -> None:
        """Honor application-level stop state."""
        ...

    def report_diagnostic_quality_degradation(
        self,
        event: str,
        error: BaseException,
    ) -> None:
        """Publish optional evidence degradation."""
        ...


class ScenarioEvidenceRuntime:
    """Delegate lifecycle while owning one private TensorBoard writer."""

    def __init__(
        self,
        *,
        delegate: EvaluationRuntimeDelegate,
        evidence_root: Path,
        cancel_requested: Callable[[], bool],
        writer_factory: SummaryWriterFactory = build_tensorboard_writer,
    ) -> None:
        self._delegate = delegate
        self._evidence_root = Path(evidence_root).resolve()
        self._cancel_requested = cancel_requested
        tensorboard_root = self._evidence_root / "tensorboard"
        tensorboard_root.mkdir()
        self._writer = writer_factory(log_dir=str(tensorboard_root))
        self._closed = False

    def get_manifest(self) -> ModelManifest:
        """Return the active model manifest without mutating selection."""

        return self._delegate.get_manifest()

    def get_tensorboard_writer(self) -> TensorboardWriter:
        """Return the scenario-owned writer."""

        return self._writer

    def get_resolved_tensorboard_log_dir(self) -> str:
        """Return the scenario root used by the trace and descriptor recorder."""

        return str(self._evidence_root)

    @staticmethod
    def get_evaluation_scenario_index() -> int:
        """Select the only scenario in the isolated manager."""

        return 0

    def check_pause(self) -> None:
        """Check cancellation before delegating application pause state."""

        self._check_cancelled()
        self._delegate.check_pause()

    def check_stop(self) -> None:
        """Check library cancellation before delegating application stop state."""

        self._check_cancelled()
        self._delegate.check_stop()

    def report_diagnostic_quality_degradation(
        self,
        event: str,
        error: BaseException,
    ) -> None:
        """Forward optional evidence degradation to the application runtime."""

        self._delegate.report_diagnostic_quality_degradation(event, error)

    def close(self) -> None:
        """Flush and close the private writer exactly once."""

        if self._closed:
            return
        self._closed = True
        flush_error: BaseException | None = None
        try:
            self._writer.flush()
        except BaseException as error:
            flush_error = error
        try:
            self._writer.close()
        except BaseException as close_error:
            if flush_error is not None:
                raise BaseExceptionGroup(
                    "Scenario evidence writer flush and close both failed.",
                    [flush_error, close_error],
                ) from close_error
            raise
        if flush_error is not None:
            raise flush_error

    def _check_cancelled(self) -> None:
        if self._cancel_requested():
            raise EvaluationSweepCancelled("Evaluation-library sweep was cancelled.")


__all__ = ["EvaluationRuntimeDelegate", "ScenarioEvidenceRuntime"]
