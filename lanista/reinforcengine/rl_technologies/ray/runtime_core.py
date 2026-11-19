# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared Ray RLlib runtime lifecycle for model plugins."""

from __future__ import annotations

import logging
import os
from typing import Protocol, cast

from lanista.foundation import system as System
from lanista.foundation.configuration import AppConfiguration
from lanista.foundation.timing import Stopper
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
    ExecutionContext,
    ExecutionLeaseFactory,
    RLModelState,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_technologies.ray.evaluation_adapter import (
    RayScenarioEvaluationAdapter,
)
from lanista.reinforcengine.rl_technologies.runtime_cleanup import close_framework_logic
from lanista.reinforcengine.rl_technologies.training_device import ray_gpu_count
from lanista.reinforcengine.rl_technologies.runtime_contracts import (
    EnvironmentFactory,
    FrameworkLogic,
    FrameworkLogicLoggerContext,
)
from lanista.reinforcengine.utils.logger_integrity_verifier import (
    LoggerIntegrityError,
    LoggerIntegrityVerifier,
)
from lanista.reinforcengine.utils.scenarios.route_entity_contracts import (
    RouteEntitiesProvider,
)
from lanista.reinforcengine.utils.scenarios.scenario_runtime import (
    build_scenario_manager,
    ensure_sumo_home,
)

APP_CONFIGURATION = AppConfiguration()


class _RayRuntimeModule(Protocol):
    """Define the typed boundary for ray runtime module."""

    def init(
        self,
        *,
        local_mode: bool,
        num_cpus: int,
        num_gpus: int,
        logging_level: int,
        log_to_driver: bool,
        include_dashboard: bool,
    ) -> object:
        """Return the init value provided by this ray runtime module."""
        ...


class RayRuntimeCore(BaseModelRuntime):
    """Own Ray process/runtime orchestration and delegate model-specific logic."""

    framework_uuid = "ray"
    environment_factory: EnvironmentFactory
    route_entities_provider: RouteEntitiesProvider | None = None

    def __init__(
        self,
        model_manifest: ModelManifest,
        *,
        execution_lease_factory: ExecutionLeaseFactory | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the ray runtime core and its owned state."""
        super().__init__(
            model_manifest,
            execution_lease_factory=execution_lease_factory,
            **kwargs,
        )
        self._ray_logger: logging.Logger | None = None
        self._cpu_count = System.get_thread_count()
        self._scenario_manager = build_scenario_manager(
            model_manifest,
            self._logger,
            route_entities_provider=self.route_entities_provider,
        )
        self._active_framework_logic: FrameworkLogic | None = None
        self._framework_cleanup_completed_ids: set[int] = set()
        # Candidate retirement can overlap a replacement start, so only a
        # worker that successfully initialized Ray may own global shutdown.
        self._ray_initialized = False
        ensure_sumo_home(self._logger)
        self._logger.info(
            f"Ray execution configured with {self._cpu_count} CPU threads for rollout workers."
        )

    async def initialize_async(self) -> RayRuntimeCore:
        """Validate the candidate without touching process-global Ray state."""

        self._validate_running_profile()
        return self

    def _initialize_sync(self) -> RayRuntimeCore:
        """Start process-global Ray for the current lease-owned worker."""

        stopper = Stopper()
        stopper.startTime()
        os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"

        import ray
        from ray import rllib, tune

        self._configure_ray_loggers()
        self._validate_running_profile()
        self._ray_initialized = False
        try:
            ray.shutdown()
            self._logger.debug("Removed previous Ray instance.")
        except Exception as error:
            self._logger.warning(f"Failed to shut down the previous Ray instance: {error}.")

        try:
            # From this point the worker owns any partial Ray process state.
            # A failed rollback must remain visible to the final cleanup pass.
            self._ray_initialized = True
            context = self._initialize_ray_context(ray)
        except BaseException:
            # Ray may leave subprocesses behind even when init raises. This
            # worker owns the attempted global transition, so it must finish
            # rollback before the execution lease is released.
            try:
                ray.shutdown()
            except Exception as cleanup_error:
                self._logger.warning(
                    f"Failed to clean up a partial Ray initialization: {cleanup_error}."
                )
            else:
                self._ray_initialized = False
            raise
        self._logger.info(ray.available_resources())
        self._verify_logger_integrity()
        dashboard_url = getattr(context, "dashboard_url", None)
        if dashboard_url:
            self._logger.info(dashboard_url)
        else:
            self._logger.info("Ray dashboard unavailable.")
        _ = rllib, tune

        hours, minutes, seconds = Stopper.formatTime(stopper.getElapsedTime())
        self._logger.info("Ray ONLINE")
        self._logger.info(
            f"Ray initialized in {hours} h {minutes} m {seconds} s with {self._cpu_count} CPUs."
        )
        return self

    def _initialize_ray_context(self, ray_module: _RayRuntimeModule) -> object:
        """Initialize the ray context owned by this ray runtime core."""
        try:
            return ray_module.init(
                local_mode=False,
                num_cpus=self._cpu_count,
                num_gpus=ray_gpu_count(),
                logging_level=logging.INFO,
                log_to_driver=False,
                include_dashboard=True,
            )
        except Exception as error:
            if not self._should_retry_without_dashboard(error):
                raise
            self._logger.warning(
                "Ray dashboard could not be started "
                f"({type(error).__name__}: {error}); retrying without it."
            )
            return ray_module.init(
                local_mode=False,
                num_cpus=self._cpu_count,
                num_gpus=ray_gpu_count(),
                logging_level=logging.INFO,
                log_to_driver=False,
                include_dashboard=False,
            )

    @staticmethod
    def _should_retry_without_dashboard(error: Exception) -> bool:
        """Return whether retry without dashboard applies to this ray runtime core."""
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "cannot include dashboard",
                "dashboard dependencies",
                "missing packages",
                "aiohttp_cors",
            )
        )

    def _validate_running_profile(self) -> None:
        """Validate the running profile required by this ray runtime core."""
        profile = self._model_manifest.get_configuration().algorithm.running_profile
        if profile not in {"single", "multi"}:
            raise ValueError(f"Invalid Ray profile: {profile}")

    def _configure_ray_loggers(self) -> None:
        """Configure the ray loggers owned by this ray runtime core."""
        counter = 0
        for name in logging.root.manager.loggerDict:
            if name.startswith("ray."):
                self._configure_ray_logger(name)
                counter += 1
        self._configure_ray_logger("ray")
        self._ray_logger = logging.getLogger("ray")
        self._ray_logger.parent = self._logger
        self._ray_logger.debug(f"Custom loggers in Ray hierarchy: {counter}")

    @staticmethod
    def _configure_ray_logger(name: str) -> None:
        """Configure the ray logger owned by this ray runtime core."""
        ray_logger = logging.getLogger(name)
        ray_logger.handlers.clear()
        ray_logger.propagate = True
        ray_logger.setLevel(RayRuntimeCore._configured_log_level())

    @staticmethod
    def _configured_log_level() -> int:
        """Return the configured log level value owned by this ray runtime core."""
        try:
            return int(APP_CONFIGURATION.logs.globally.level)
        except (AttributeError, TypeError, ValueError):
            return logging.INFO

    def _verify_logger_integrity(self) -> None:
        """Verify the logger integrity owned by this ray runtime core."""
        verifier = LoggerIntegrityVerifier(
            parent_logger_name=self._logger.name,
            child_logger_name="logger_integrity_verifier",
            child_logger_level=self._configured_log_level(),
        )
        APP_CONFIGURATION.logger_integrity_verifier = verifier
        self._logger.info("Logger integrity verification ONLINE.")
        try:
            verifier.verify()
        except LoggerIntegrityError as error:
            # Logger validation is diagnostic: a third-party Ray logger must not
            # make an otherwise healthy runtime unavailable.
            self._logger.error(f"Logger integrity verification FAILED: {error}")
        else:
            self._logger.info("Logger integrity verification PASSED.")

    def _new_logic(self) -> FrameworkLogic:
        """Return the new logic value owned by this ray runtime core."""
        logic = self.environment_factory.create(
            self,
            FrameworkLogicLoggerContext(
                parent_logger_name=self._logger.name,
                child_logger_name="ray_sim_logic",
                child_logger_level=self._configured_log_level(),
            ),
        )
        self._active_framework_logic = logic
        self._framework_cleanup_completed_ids = set()
        return logic

    def _train(self) -> None:
        """Train the operation owned by this ray runtime core."""
        self._initialize_sync()
        self._scenario_manager.generate()  # type: ignore[no-untyped-call]
        logic = self._new_logic()
        selected_session_name = getattr(
            self._model_manifest.get_configuration(),
            "selected_default_model_name",
            None,
        )
        source_session_name, source_checkpoint_name = self.get_training_checkpoint_source()
        if source_session_name and source_checkpoint_name:
            logic.load(
                model_name=source_session_name,
                checkpoint_name=source_checkpoint_name,
            )
            if source_session_name != selected_session_name:
                logic.build()
                setattr(logic, "_base_step_offset", 0)  # noqa: B010
        else:
            logic.build()

        try:
            logic.train(self._scenario_manager, self._training_progress_reporter())
            logic.save()
        except InterruptedError as error:
            self._handle_interrupted_operation(
                logic,
                error,
                operation_name="Training",
            )

    def _evaluate(self) -> None:
        """Evaluate the operation owned by this ray runtime core."""
        selection = self._require_selected_checkpoint("Evaluation")
        self._initialize_sync()
        self._scenario_manager.generate()  # type: ignore[no-untyped-call]
        logic = self._new_logic()
        try:
            logic.load(
                model_name=selection.session_name,
                checkpoint_name=selection.checkpoint_name,
            )
            logic.deploy(
                self._scenario_manager,
                purpose="evaluation",
                repetitions=self.get_evaluation_repetition_count(),
                visualization=self.get_evaluation_visualization(),
                realtime=self.evaluation_uses_realtime_pacing(),
            )
        except InterruptedError as error:
            self._handle_interrupted_operation(
                logic,
                error,
                operation_name="Evaluation",
            )

    def _evaluate_library(
        self,
        execution: LibraryEvaluationExecution,
        context: ExecutionContext,
    ) -> EvaluationSweepResult:
        """Run every held-out scenario through one cached inference RLModule."""

        self._initialize_sync()
        logic = self._new_logic()
        adapter = RayScenarioEvaluationAdapter(
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
        """Infer the operation owned by this ray runtime core."""
        selection = self._require_selected_checkpoint("Inference")
        self._initialize_sync()
        self._scenario_manager.generate()  # type: ignore[no-untyped-call]
        logic = self._new_logic()
        try:
            logic.load(
                model_name=selection.session_name,
                checkpoint_name=selection.checkpoint_name,
            )
            logic.deploy(self._scenario_manager, purpose="deployment")
        except InterruptedError as error:
            self._handle_interrupted_operation(
                logic,
                error,
                operation_name="Inference",
            )

    def _handle_interrupted_operation(
        self,
        logic: FrameworkLogic,
        error: InterruptedError,
        *,
        operation_name: str,
    ) -> None:
        """Handle the interrupted operation workflow for this ray runtime core."""
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
                total=None,
                progress=0,
                status=RLModelState.IDLE,
                additional_status_message=message,
            )

    def _cleanup_execution_resources(self) -> None:
        """Clean up the execution resources owned by this ray runtime core."""
        logic = self._active_framework_logic
        owns_ray_runtime = getattr(self, "_ray_initialized", False)
        cleanup_error: Exception | None = None
        if logic is not None:
            try:
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
            except Exception as error:
                cleanup_error = error
            else:
                self._active_framework_logic = None
                self._framework_cleanup_completed_ids = set()

        if owns_ray_runtime:
            try:
                import ray

                ray.shutdown()
            except Exception as error:
                cleanup_error = cleanup_error or error
            else:
                self._ray_initialized = False
        if cleanup_error is not None:
            raise RuntimeError("Ray runtime cleanup failed.") from cleanup_error
