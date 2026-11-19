# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Adapt one framework inference executor to the shared sweep coordinator."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from lanista.reinforcengine.rl_model.evaluation.coordinator import (
    EvaluationScenarioOutcome,
    ScenarioEvaluationRequest,
)
from lanista.reinforcengine.rl_model.evaluation.evidence_runtime import (
    EvaluationRuntimeDelegate,
    ScenarioEvidenceRuntime,
)
from lanista.reinforcengine.rl_model.runtime.tensorboard_writer import (
    SummaryWriterFactory,
    build_tensorboard_writer,
)
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import (
    ScenarioManager,
)


class ScenarioInferenceExecutor(Protocol):
    """Run one explicit checkpoint through a framework's native inference path."""

    def evaluate_scenario_inference(
        self,
        *,
        scenario_manager: ScenarioManager,
        checkpoint_path: Path,
        runtime: ScenarioEvidenceRuntime,
        deterministic: bool,
        repetitions: int,
    ) -> str:
        """Return the terminal outcome after closing the local environment."""
        ...


class FrameworkScenarioEvaluationAdapter:
    """Isolate scenario inventory and evidence around a framework executor."""

    def __init__(
        self,
        *,
        executor: ScenarioInferenceExecutor,
        scenario_manager: ScenarioManager,
        runtime: EvaluationRuntimeDelegate,
        writer_factory: SummaryWriterFactory = build_tensorboard_writer,
    ) -> None:
        self._executor = executor
        self._scenario_manager = scenario_manager
        self._runtime = runtime
        self._writer_factory = writer_factory

    def evaluate_scenario(
        self,
        request: ScenarioEvaluationRequest,
    ) -> EvaluationScenarioOutcome:
        """Run native inference in one held-out scenario and evidence scope."""

        scenario_manager = self._scenario_manager.clone_for_evaluation_scenario(
            request.scenario_root
        )
        runtime = ScenarioEvidenceRuntime(
            delegate=self._runtime,
            evidence_root=request.evidence_root,
            cancel_requested=request.cancel_requested,
            writer_factory=self._writer_factory,
        )
        try:
            terminal_outcome = self._executor.evaluate_scenario_inference(
                scenario_manager=scenario_manager,
                checkpoint_path=request.checkpoint_path,
                runtime=runtime,
                deterministic=request.deterministic_inference,
                repetitions=request.repetitions,
            )
        except BaseException as inference_error:
            try:
                runtime.close()
            except BaseException as close_error:
                inference_error.add_note(
                    "Scenario evidence cleanup also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            raise
        else:
            runtime.close()
        return EvaluationScenarioOutcome(terminal_outcome=terminal_outcome)


__all__ = ["FrameworkScenarioEvaluationAdapter", "ScenarioInferenceExecutor"]
