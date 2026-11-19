# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared Stable-Baselines3 runtime lifecycle for model plugins."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, cast

from lanista.foundation.configuration import AppConfiguration
from lanista.reinforcengine.rl_model.evaluation.coordinator import (
    EvaluationSweepCoordinator,
    EvaluationSweepResult,
)
from lanista.reinforcengine.rl_model.evaluation.framework_adapter import (
    ScenarioInferenceExecutor,
)
from lanista.reinforcengine.rl_model.evaluation.runtime_execution import (
    LibraryEvaluationExecution,
)
from lanista.reinforcengine.rl_model.execution_contracts import (
    EvaluationVisualization,
    ExecutionContext,
    ExecutionLeaseFactory,
    RLModelState,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_technologies.runtime_cleanup import close_framework_logic
from lanista.reinforcengine.rl_technologies.runtime_contracts import (
    DeploymentPurpose,
    EnvironmentFactory,
    FrameworkLogic,
    FrameworkLogicLoggerContext,
)
from lanista.reinforcengine.rl_technologies.sb3.evaluation_adapter import (
    SB3ScenarioEvaluationAdapter,
)
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import (
    ScenarioManager,
)
from lanista.reinforcengine.utils.scenarios.route_entity_contracts import (
    RouteEntitiesProvider,
)
from lanista.reinforcengine.utils.scenarios.scenario_runtime import (
    build_scenario_manager,
    ensure_sumo_home,
)

APP_CONFIGURATION = AppConfiguration()


@dataclass(frozen=True, slots=True)
class SB3EvaluationStrategy:
    """Configure evaluation choices while the runtime owns their lifecycle."""

    deployment_purpose: DeploymentPurpose = "evaluation"


class _SpecializedEvaluationLogic(Protocol):
    """SB3-only hooks used by an explicitly configured evaluation strategy."""

    def deploy(
        self,
        scenario_manager: ScenarioManager,
        *,
        purpose: DeploymentPurpose = "deployment",
        repetitions: int = 1,
        visualization: EvaluationVisualization = "gui",
        realtime: bool | None = None,
    ) -> None:
        """Return the deploy value provided by this specialized evaluation logic."""
        ...


class SB3RuntimeCore(BaseModelRuntime):
    """Own SB3 orchestration while model plugins supply environment logic."""

    framework_uuid = "sb3"
    environment_factory: EnvironmentFactory
    route_entities_provider: RouteEntitiesProvider | None = None
    evaluation_strategy = SB3EvaluationStrategy()

    def __init__(
        self,
        model_manifest: ModelManifest,
        *,
        execution_lease_factory: ExecutionLeaseFactory | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the sb3 runtime core and its owned state."""
        super().__init__(
            model_manifest,
            execution_lease_factory=execution_lease_factory,
            **kwargs,
        )
        self._scenario_manager = build_scenario_manager(
            model_manifest,
            self._logger,
            route_entities_provider=self.route_entities_provider,
        )
        self._active_framework_logic: FrameworkLogic | None = None
        self._framework_cleanup_completed_ids: set[int] = set()
        ensure_sumo_home(self._logger)

    async def initialize_async(self) -> SB3RuntimeCore:
        """Initialize the async owned by this sb3 runtime core."""
        return self

    def _new_logic(self) -> FrameworkLogic:
        """Return the new logic value owned by this sb3 runtime core."""
        logic = self.environment_factory.create(
            self,
            self._logic_logger_context(),
        )
        self._active_framework_logic = logic
        self._framework_cleanup_completed_ids = set()
        return logic

    def _logic_logger_context(self) -> FrameworkLogicLoggerContext:
        """Return the logic logger context value owned by this sb3 runtime core."""
        return FrameworkLogicLoggerContext(
            parent_logger_name=self._logger.name,
            child_logger_name="stable_baselines_sim_logic",
            child_logger_level=self._configured_log_level(),
        )

    @staticmethod
    def _configured_log_level() -> int:
        """Return the configured log level value owned by this sb3 runtime core."""
        try:
            return int(APP_CONFIGURATION.logs.globally.level)
        except (AttributeError, TypeError, ValueError):
            return logging.INFO

    def _train(self) -> None:
        """Train the operation owned by this sb3 runtime core."""
        self._scenario_manager.generate()  # type: ignore[no-untyped-call]
        logic = self._new_logic()
        logic.build()
        try:
            logic.train(self._scenario_manager, self._training_progress_reporter())
            logic.save()
        except InterruptedError as exc:
            self._handle_interrupted_operation(
                logic,
                exc,
                operation_name="Training",
            )

    def _evaluate(self) -> None:
        """Evaluate the operation owned by this sb3 runtime core."""
        selection = self._require_selected_checkpoint("Evaluation")
        self._scenario_manager.generate()  # type: ignore[no-untyped-call]
        logic = self._new_logic()
        try:
            self._run_evaluation_strategy(
                logic,
                selected_session_name=selection.session_name,
                selected_checkpoint_name=selection.checkpoint_name,
            )
        except InterruptedError as exc:
            self._handle_interrupted_operation(
                logic,
                exc,
                operation_name="Evaluation",
            )

    def _evaluate_library(
        self,
        execution: LibraryEvaluationExecution,
        context: ExecutionContext,
    ) -> EvaluationSweepResult:
        """Run every held-out scenario through one cached SB3 policy."""

        logic = self._new_logic()
        adapter = SB3ScenarioEvaluationAdapter(
            executor=cast(ScenarioInferenceExecutor, logic),
            scenario_manager=self._scenario_manager,
            runtime=self,
        )
        return EvaluationSweepCoordinator(
            execution.library_storage,
            utc_now=execution.utc_now,
        ).run(
            execution.sweep_request,
            adapter,
            cancel_requested=lambda: context.stop_requested,
            progress_callback=execution.progress_callback,
        )

    def _infer(self) -> None:
        """Infer the operation owned by this sb3 runtime core."""
        selection = self._require_selected_checkpoint("Inference")
        self._scenario_manager.generate()  # type: ignore[no-untyped-call]
        logic = self._new_logic()
        try:
            logic.load(
                model_name=selection.session_name,
                checkpoint_name=selection.checkpoint_name,
            )
            logic.deploy(self._scenario_manager, purpose="deployment")
        except InterruptedError as exc:
            self._handle_interrupted_operation(
                logic,
                exc,
                operation_name="Inference",
            )

    def _run_evaluation_strategy(
        self,
        logic: FrameworkLogic,
        *,
        selected_session_name: str | None,
        selected_checkpoint_name: str | None,
    ) -> None:
        """Run the evaluation strategy owned by this sb3 runtime core."""
        strategy = self.evaluation_strategy
        logic.load(
            model_name=selected_session_name,
            checkpoint_name=selected_checkpoint_name,
        )
        cast(_SpecializedEvaluationLogic, logic).deploy(
            self._scenario_manager,
            purpose=strategy.deployment_purpose,
            repetitions=self.get_evaluation_repetition_count(),
            visualization=self.get_evaluation_visualization(),
            realtime=self.evaluation_uses_realtime_pacing(),
        )

    def _handle_interrupted_operation(
        self,
        logic: FrameworkLogic,
        error: InterruptedError,
        *,
        operation_name: str,
    ) -> None:
        """Handle the interrupted operation workflow for this sb3 runtime core."""
        training = operation_name == "Training"
        self._logger.error(f"{operation_name} interrupted: {error}")
        self._logger.warning(f"{operation_name} stopped by a user request or framework stop event.")
        manual_checkpoint_name = None
        if training:
            try:
                manual_checkpoint_name = logic.save(checkpoint=True, manual=True)
            except Exception as save_error:
                self._logger.warning(
                    f"Manual checkpoint save failed during stop handling: {save_error}"
                )
        if self._progress_callback:
            message = f"{operation_name} stopped by user"
            if manual_checkpoint_name:
                message += f". Manual checkpoint saved: {manual_checkpoint_name}"
            self._update_progress(
                total=None if training else -1,
                progress=0 if training else -1,
                status=RLModelState.IDLE,
                additional_status_message=message,
            )

    def _cleanup_execution_resources(self) -> None:
        """Clean up the execution resources owned by this sb3 runtime core."""
        logic = self._active_framework_logic
        if logic is None:
            return
        completed_ids: set[int] = getattr(
            self,
            "_framework_cleanup_completed_ids",
            set(),
        )
        self._framework_cleanup_completed_ids = completed_ids
        close_framework_logic(
            logic,
            completed_resource_ids=completed_ids,
        )
        self._active_framework_logic = None
        self._framework_cleanup_completed_ids = set()
