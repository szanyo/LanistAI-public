# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Define the reinforcement-learning model runtime."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Event, RLock, Thread
from time import monotonic
from typing import cast

from lanista.foundation.configuration import AppConfiguration
from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.application_errors import ApplicationValidationError
from lanista.reinforcengine.rl_model.checkpoints.runtime_source_resolver import (
    TrainingCheckpointSourceResolver,
)
from lanista.reinforcengine.rl_model.evaluation.contracts import EvaluationSweepStatus
from lanista.reinforcengine.rl_model.evaluation.coordinator import (
    EvaluationSweepResult,
    fail_report_publication,
)
from lanista.reinforcengine.rl_model.evaluation.runtime_execution import (
    LibraryEvaluationExecution,
    require_successful_sweep,
)
from lanista.reinforcengine.rl_model.execution_contracts import (
    ACTIVE_EXECUTION_STATES,
    EvaluationPacing,
    EvaluationVisualization,
    ExecutionContext,
    ExecutionLease,
    ExecutionLeaseFactory,
    ExecutionSnapshot,
    ExecutionStopTimeoutError,
    InvalidExecutionStateError,
    RLModelState,
)
from lanista.reinforcengine.rl_model.execution_coordinator import (
    ExecutionCoordinator,
    ExecutionHandle,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.runtime.requests import (
    RuntimeSelectionProvider,
    RuntimeSelectionSnapshot,
)
from lanista.reinforcengine.rl_model.runtime.resource_disposer import (
    RuntimeResourceDisposer,
)
from lanista.reinforcengine.rl_model.runtime.tensorboard_fallback import (
    NoopSummaryWriter,
)
from lanista.reinforcengine.rl_model.runtime.tensorboard_layout_service import (
    TensorboardLayoutMode,
)
from lanista.reinforcengine.rl_model.runtime.tensorboard_writer import (
    TensorboardWriter,
    TensorboardWriterOwner,
)
from lanista.reinforcengine.utils.scenarios.scenario_selection import (
    resolve_scenario_index,
)

APP_CONFIGURATION = AppConfiguration()
ProgressCallback = Callable[
    [float | None, float | None, RLModelState, str],
    None,
]


class RLModelRunningModes(Enum):
    """Enumerate the execution modes supported by an RL model."""

    TRAINING = "Training"
    INFERENCE = "Inference"
    EVALUATION = "Evaluation"


@dataclass
class ModelExecutionUpdate:
    """Carry the data required for model execution update."""

    total: float | None
    progress: float | None
    status: RLModelState
    message: str = ""
    run_id: str | None = None


class _UnmanagedExecutionLeaseFactory:
    """Fail closed when a runtime plugin discarded its lifecycle dependency."""

    def acquire_execution_lease(self, run_id: str) -> ExecutionLease:
        """Return the acquire execution lease value provided by this unmanaged execution lease
        factory.
        """
        raise InvalidExecutionStateError(
            "The model runtime has no managed execution lifecycle.",
            state=RLModelState.IDLE,
            run_id=run_id,
        )


class BaseModelRuntime(LoggerInterface):
    """Reusable model runtime whose worker lifecycle is externally owned."""

    _TRAINING_WARMUP_HEARTBEAT_SECONDS = 2.0

    def __init__(
        self,
        model_manifest: ModelManifest,
        *,
        execution_lease_factory: ExecutionLeaseFactory | None = None,
        runtime_selection_provider: RuntimeSelectionProvider | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the model with optional keyword arguments."""
        LoggerInterface.__init__(self, **kwargs)
        self._model_manifest: ModelManifest = model_manifest
        self._runtime_selection_provider = runtime_selection_provider
        self._training_checkpoint_source_resolver = TrainingCheckpointSourceResolver()
        self._running_mode: RLModelRunningModes = RLModelRunningModes.TRAINING
        self._evaluation_visualization_override: EvaluationVisualization | None = None
        self._evaluation_pacing_override: EvaluationPacing | None = None
        self._evaluation_scenario_index_override: int | None = None
        self._last_evaluation_sweep_result: EvaluationSweepResult | None = None
        self._tensorboard_writer_owner = TensorboardWriterOwner(
            configuration_provider=self._model_manifest.get_configuration,
            layout_mode_provider=self._tensorboard_layout_mode,
            logger_provider=lambda: self._logger,
        )
        self._inference_writer = NoopSummaryWriter()
        self._progress_callback: ProgressCallback | None = None
        self._last_execution_update: ModelExecutionUpdate | None = None
        self._progress_callback_lock = RLock()
        self._training_warmup_lock = RLock()
        self._training_warmup_complete = Event()
        self._initialization_cancel_event = Event()
        self._execution_lifecycle_managed = execution_lease_factory is not None
        degraded_state_reporter = getattr(
            execution_lease_factory,
            "mark_degraded",
            None,
        )
        self._degraded_state_reporter = (
            degraded_state_reporter if callable(degraded_state_reporter) else None
        )
        self._execution_coordinator = ExecutionCoordinator(
            on_transition=self._on_execution_transition,
            execution_lease_factory=(execution_lease_factory or _UnmanagedExecutionLeaseFactory()),
            on_cleanup_failure=self._report_execution_cleanup_failure,
        )
        self._resource_disposer = RuntimeResourceDisposer(
            execution=self._execution_coordinator,
            writer=self._tensorboard_writer_owner,
            framework_cleanup=self._cleanup_execution_resources,
        )

    def get_resolved_tensorboard_log_dir(self) -> str | None:
        """Return the writer-owned run directory for the active mode."""
        if self._running_mode is RLModelRunningModes.INFERENCE:
            return None
        with self._resource_disposer.resource_creation_guard():
            return self._tensorboard_writer_owner.resolve_log_dir()

    def report_diagnostic_quality_degradation(
        self,
        event: str,
        error: BaseException,
    ) -> None:
        """Report an optional diagnostic failure without changing run state."""

        self._logger.warning(
            f"event={event} quality=degraded "
            f"error_type={type(error).__name__} error={error}"
        )

    def get_tensorboard_step_floor(self) -> int:
        """Return the recovered step floor for the active logical session."""

        if self._running_mode is RLModelRunningModes.INFERENCE:
            return 0
        with self._resource_disposer.resource_creation_guard():
            self._tensorboard_writer_owner.resolve_log_dir()
            return self._tensorboard_writer_owner.step_floor()

    def _tensorboard_layout_mode(self) -> TensorboardLayoutMode:
        """Return the tensorboard layout mode value owned by this base model runtime."""
        if self._running_mode is RLModelRunningModes.EVALUATION:
            return TensorboardLayoutMode.EVALUATION
        return TensorboardLayoutMode.TRAINING

    def get_evaluation_repetition_count(self) -> int:
        """Return the framework-neutral number of evaluation episodes."""

        configuration = self._model_manifest.get_configuration()
        runtime_configuration = getattr(configuration, "runtime", None)
        evaluation_configuration = getattr(runtime_configuration, "evaluation", None)
        raw_count = getattr(evaluation_configuration, "n_eval_episodes", 1)
        try:
            repetition_count = int(raw_count)
        except (TypeError, ValueError) as error:
            raise ApplicationValidationError(
                "Evaluation n_eval_episodes must be a positive integer."
            ) from error
        if repetition_count < 1:
            raise ApplicationValidationError(
                "Evaluation n_eval_episodes must be a positive integer."
            )
        # The embedded native GUI owns one process image per evaluation.  A
        # configured repetition count remains useful for headless statistics,
        # but GUI evaluation always means one run of the selected scenario.
        if self.get_evaluation_visualization() == "gui":
            return 1
        return repetition_count

    def get_evaluation_visualization(self) -> EvaluationVisualization:
        """Return whether evaluation renders an embedded GUI."""

        override = getattr(self, "_evaluation_visualization_override", None)
        if override is not None:
            return cast(EvaluationVisualization, override)
        configuration = self._model_manifest.get_configuration()
        runtime_configuration = getattr(configuration, "runtime", None)
        evaluation_configuration = getattr(runtime_configuration, "evaluation", None)
        visualization = (
            str(getattr(evaluation_configuration, "visualization", "gui") or "gui").strip().lower()
        )
        if visualization not in {"gui", "headless"}:
            raise ApplicationValidationError(
                "Evaluation visualization must be either 'gui' or 'headless'."
            )
        return cast(EvaluationVisualization, visualization)

    def get_evaluation_pacing(self) -> EvaluationPacing:
        """Return the scheduler pacing selected for Evaluation."""

        override = getattr(self, "_evaluation_pacing_override", None)
        if override is not None:
            return cast(EvaluationPacing, override)
        configuration = self._model_manifest.get_configuration()
        runtime_configuration = getattr(configuration, "runtime", None)
        evaluation_configuration = getattr(runtime_configuration, "evaluation", None)
        pacing = (
            str(
                getattr(evaluation_configuration, "pacing", "maximum-throughput")
                or "maximum-throughput"
            )
            .strip()
            .lower()
        )
        if pacing not in {"maximum-throughput", "realtime"}:
            raise ApplicationValidationError(
                "Evaluation pacing must be either 'maximum-throughput' or 'realtime'."
            )
        return cast(EvaluationPacing, pacing)

    def evaluation_uses_realtime_pacing(self) -> bool:
        """Return whether scheduler delay is active for this Evaluation."""

        return (
            self.get_evaluation_visualization() == "gui"
            or self.get_evaluation_pacing() == "realtime"
        )

    def get_available_evaluation_scenario_indices(self) -> tuple[int, ...]:
        """Return the scenario indices exposed by the active runtime inventory."""

        scenario_manager = getattr(self, "_scenario_manager", None)
        total_provider = getattr(
            scenario_manager,
            "get_total_number_of_scenarios",
            None,
        )
        if not callable(total_provider):
            return ()
        try:
            total = int(total_provider())
        except (TypeError, ValueError) as error:
            raise ApplicationValidationError(
                "The runtime scenario inventory size must be a non-negative integer."
            ) from error
        if total < 0:
            raise ApplicationValidationError(
                "The runtime scenario inventory size must be a non-negative integer."
            )
        return tuple(range(total))

    def get_evaluation_scenario_index(self) -> int:
        """Return the transient override or configured Evaluation scenario index."""

        override = getattr(self, "_evaluation_scenario_index_override", None)
        scenario_index = (
            override
            if override is not None
            else resolve_scenario_index(
                self._model_manifest.get_configuration(),
                purpose="evaluation",
                default_index=0,
            )
        )
        available = self.get_available_evaluation_scenario_indices()
        if available and scenario_index not in available:
            raise ApplicationValidationError(
                f"Evaluation scenario index {scenario_index} is outside the available "
                f"range 0..{available[-1]}."
            )
        if scenario_index < 0:
            raise ApplicationValidationError(
                "Evaluation scenario index must be a non-negative integer."
            )
        return scenario_index

    def set_evaluation_scenario_index(self, scenario_index: int | None) -> None:
        """Select or clear the transient Evaluation scenario while idle."""

        if scenario_index is not None and (
            isinstance(scenario_index, bool) or not isinstance(scenario_index, int)
        ):
            raise ApplicationValidationError(
                "Evaluation scenario index must be a non-negative integer."
            )
        available = self.get_available_evaluation_scenario_indices()
        if scenario_index is not None and (
            scenario_index < 0 or (available and scenario_index not in available)
        ):
            range_detail = f"0..{available[-1]}" if available else "non-negative integers"
            raise ApplicationValidationError(
                f"Evaluation scenario index {scenario_index} is outside the available "
                f"range {range_detail}."
            )
        with self._resource_disposer.configuration_guard():
            snapshot = self._execution_coordinator.snapshot()
            if snapshot.state in ACTIVE_EXECUTION_STATES or snapshot.worker_alive:
                raise InvalidExecutionStateError(
                    "Cannot change the evaluation scenario during execution.",
                    state=snapshot.state,
                    run_id=snapshot.run_id,
                )
            self._evaluation_scenario_index_override = scenario_index

    def set_evaluation_visualization(
        self,
        visualization: EvaluationVisualization,
    ) -> None:
        """Select Evaluation rendering while retaining its pacing setting."""

        self.set_evaluation_profile(visualization, self.get_evaluation_pacing())

    def set_evaluation_profile(
        self,
        visualization: EvaluationVisualization,
        pacing: EvaluationPacing,
    ) -> None:
        """Atomically select Evaluation rendering and pacing while idle."""

        if visualization not in {"gui", "headless"}:
            raise ApplicationValidationError(
                "Evaluation visualization must be either 'gui' or 'headless'."
            )
        if pacing not in {"maximum-throughput", "realtime"}:
            raise ApplicationValidationError(
                "Evaluation pacing must be either 'maximum-throughput' or 'realtime'."
            )
        with self._resource_disposer.configuration_guard():
            snapshot = self._execution_coordinator.snapshot()
            if snapshot.state in ACTIVE_EXECUTION_STATES or snapshot.worker_alive:
                raise InvalidExecutionStateError(
                    "Cannot change evaluation visualization during execution.",
                    state=snapshot.state,
                    run_id=snapshot.run_id,
                )
            self._evaluation_visualization_override = visualization
            self._evaluation_pacing_override = pacing

    def _runtime_selection_snapshot(self) -> RuntimeSelectionSnapshot:
        """Snapshot the active selection once for one framework operation."""

        provider = getattr(self, "_runtime_selection_provider", None)
        if provider is not None:
            selection = provider()
            return RuntimeSelectionSnapshot(
                session_name=str(selection.session_name or "").strip() or None,
                checkpoint_name=str(selection.checkpoint_name or "").strip() or None,
            )
        configuration = self._model_manifest.get_configuration()
        return RuntimeSelectionSnapshot(
            session_name=(
                str(getattr(configuration, "selected_default_model_name", "") or "").strip()
                or None
            ),
            checkpoint_name=(
                str(
                    getattr(configuration, "selected_default_checkpoint_name", "") or ""
                ).strip()
                or None
            ),
        )

    def _require_selected_checkpoint(
        self,
        operation_name: str,
        selection: RuntimeSelectionSnapshot | None = None,
    ) -> RuntimeSelectionSnapshot:
        """Return one explicit runtime selection or fail before mutation."""

        resolved = selection or self._runtime_selection_snapshot()
        if resolved.checkpoint_name:
            return resolved
        raise ApplicationValidationError(
            f"{operation_name} requires an explicitly selected checkpoint."
        )

    def get_training_checkpoint_source(self) -> tuple[str | None, str | None]:
        """Return the session/checkpoint pair to restore before training.

        The selected session remains the save target. Forked sessions can name a
        parent checkpoint source in their session manifest.
        """
        return self._training_checkpoint_source_resolver.resolve(
            self._model_manifest.get_configuration()
        ).as_pair()

    def refresh_tensorboard_writer(self) -> None:
        """Refresh the tensorboard writer owned by this base model runtime."""
        if self._running_mode is RLModelRunningModes.INFERENCE:
            return
        with self._resource_disposer.configuration_guard():
            snapshot = self._execution_coordinator.snapshot()
            if snapshot.state in ACTIVE_EXECUTION_STATES or snapshot.worker_alive:
                raise InvalidExecutionStateError(
                    "Cannot refresh the TensorBoard writer during execution.",
                    state=snapshot.state,
                    run_id=snapshot.run_id,
                )
            self._tensorboard_writer_owner.refresh()

    async def initialize_async(self) -> "BaseModelRuntime":
        """Initialize the async owned by this base model runtime."""
        raise NotImplementedError("Subclasses should implement this method.")

    def request_initialization_cancel(self) -> None:
        """Signal a withdrawn plugin initializer to stop external side effects."""

        self._initialization_cancel_event.set()

    def initialization_cancel_requested(self) -> bool:
        """Return whether the loader withdrew this runtime candidate."""

        return self._initialization_cancel_event.is_set()

    async def wait_for_initialization_quiescence(self) -> None:
        """Wait until plugin-owned initialization workers have stopped.

        Plugins that delegate initialization to an executor, process, or other
        background worker must override this hook and join that worker. The
        loader may release its exclusive load lease only after this coroutine
        returns.
        """

    async def cancel_initialization_and_wait(
        self,
        initialization_task: "asyncio.Task[BaseModelRuntime]",
    ) -> None:
        """Cancel the initializer coroutine and join delegated initialization."""

        self.request_initialization_cancel()
        initialization_task.cancel()
        try:
            await initialization_task
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self._logger.warning(
                f"Runtime initialization stopped with [{type(error).__name__}]: {error}"
            )
        await self.wait_for_initialization_quiescence()

    def set_running_mode(self, mode: RLModelRunningModes) -> None:
        """Set the running mode (training or evaluation)."""
        with self._resource_disposer.configuration_guard():
            snapshot = self._execution_coordinator.snapshot()
            if snapshot.state in ACTIVE_EXECUTION_STATES or snapshot.worker_alive:
                raise InvalidExecutionStateError(
                    "Cannot change the running mode while an execution is active.",
                    state=snapshot.state,
                    run_id=snapshot.run_id,
                )
            self._running_mode = mode
        self._logger.debug(f"Running mode set to {mode.value}")

    def get_running_mode(self) -> RLModelRunningModes:
        """Return the runtime-owned mode of the current or most recent run."""

        return self._running_mode

    def set_progress_callback(
        self,
        callback: ProgressCallback,
    ) -> None:
        """Set the callback function for reporting progress updates.

        Args:
            callback: Function receiving total, progress, state, and a message.
        """
        with self._progress_callback_lock:
            self._progress_callback = callback
        self._logger.debug("Progress callback set successfully.")

    def clear_progress_callback(self, callback: ProgressCallback) -> bool:
        """Detach only the adapter callback that still owns the active slot."""

        with self._progress_callback_lock:
            if self._progress_callback is not callback:
                return False
            self._progress_callback = None
        self._logger.debug("Progress callback cleared successfully.")
        return True

    def get_last_execution_update(self) -> ModelExecutionUpdate | None:
        """Return the last execution update exposed by this base model runtime."""
        return self._last_execution_update

    def get_execution_snapshot(self) -> ExecutionSnapshot:
        """Return the coordinator's immutable, thread-safe state snapshot."""
        return self._execution_coordinator.snapshot()

    def has_managed_execution_lifecycle(self) -> bool:
        """Return whether construction retained the application lease factory."""

        return self._execution_lifecycle_managed

    async def start_operation(self) -> ExecutionSnapshot:
        """Run the selected operation on a newly allocated worker."""
        return await self.begin_operation().wait()

    async def start_evaluation_library(
        self,
        execution: LibraryEvaluationExecution,
    ) -> ExecutionSnapshot:
        """Run one explicit held-out library through the managed lifecycle."""

        return await self.begin_evaluation_library(execution).wait()

    def begin_evaluation_library(
        self,
        execution: LibraryEvaluationExecution,
    ) -> ExecutionHandle:
        """Start a headless library sweep without changing interactive Evaluation."""

        self._validate_library_execution(execution)
        return self._begin_managed_operation(
            lambda context: self._execute_evaluation_library(execution, context)
        )

    def get_last_evaluation_sweep_result(self) -> EvaluationSweepResult | None:
        """Return the most recent persisted library result for this runtime."""

        return self._last_evaluation_sweep_result

    def begin_operation(
        self,
        *,
        prepare: Callable[[], None] | None = None,
    ) -> ExecutionHandle:
        """Start a managed worker synchronously and return its awaitable handle."""

        return self._begin_managed_operation(
            self._execute_operation,
            prepare=prepare,
        )

    def _begin_managed_operation(
        self,
        operation: Callable[[ExecutionContext], None],
        *,
        prepare: Callable[[], None] | None = None,
    ) -> ExecutionHandle:
        """Start one runtime-owned operation through the shared lifecycle."""

        with self._resource_disposer.start_guard():
            handle = self._execution_coordinator.start(
                operation,
                cleanup=self._cleanup_execution_run,
                prepare=lambda context: self._prepare_execution_run(
                    context,
                    prepare,
                ),
            )
        return handle

    def _prepare_execution_run(
        self,
        _context: ExecutionContext,
        prepare: Callable[[], None] | None,
    ) -> None:
        """Reset per-run cleanup ownership before external preparation."""

        self._resource_disposer.prepare_execution_run()
        self._training_warmup_complete.clear()
        if prepare is not None:
            prepare()

    def _execute_operation(self, context: ExecutionContext) -> None:
        """Execute the operation owned by this base model runtime."""
        if not self._has_progress_callback():
            self._logger.warning("No progress callback set. Progress updates will not be reported.")

        context.check_stop()
        heartbeat: Thread | None = None
        if self._running_mode is RLModelRunningModes.TRAINING and self._has_progress_callback():
            heartbeat = Thread(
                target=self._publish_training_warmup_heartbeat,
                args=(context.run_id,),
                name=f"lanistai-training-warmup-{context.run_id[:8]}",
                daemon=True,
            )
            heartbeat.start()
        try:
            if self._running_mode == RLModelRunningModes.TRAINING:
                self._train()
            elif self._running_mode == RLModelRunningModes.EVALUATION:
                self._evaluate()
            elif self._running_mode == RLModelRunningModes.INFERENCE:
                self._infer()
            else:
                raise ValueError(f"Invalid running mode: {self._running_mode}")
        finally:
            self._training_warmup_complete.set()
            if heartbeat is not None:
                heartbeat.join(timeout=0.2)

    def _execute_evaluation_library(
        self,
        execution: LibraryEvaluationExecution,
        context: ExecutionContext,
    ) -> None:
        """Revalidate selection and publish the terminal sweep result."""

        self._validate_library_execution(execution)
        result = self._evaluate_library(execution, context)
        if (
            execution.sweep_request.run_configuration is not None
            and result.descriptor.status is EvaluationSweepStatus.COMPLETED
        ):
            publisher = execution.report_publisher
            if publisher is None:
                raise RuntimeError("Grouped evaluation lacks a report publisher.")
            try:
                publisher(result.run_root)
            except Exception as error:
                result = fail_report_publication(result, str(error))
        self._last_evaluation_sweep_result = result
        require_successful_sweep(result)

    def _validate_library_execution(
        self,
        execution: LibraryEvaluationExecution,
    ) -> None:
        """Bind a sweep to this framework and the active checkpoint selection."""

        request = execution.sweep_request
        framework_uuid = str(getattr(self, "framework_uuid", "") or "").lower()
        if request.framework != framework_uuid:
            raise ApplicationValidationError(
                f"Evaluation sweep framework {request.framework!r} does not match "
                f"the active {framework_uuid or '<unknown>'!r} runtime."
            )
        selection = self._require_selected_checkpoint("Evaluation-library sweep")
        if selection.session_name != request.session_id:
            raise ApplicationValidationError(
                "Evaluation sweep session does not match the active runtime selection."
            )
        if selection.checkpoint_name != request.checkpoint_id:
            raise ApplicationValidationError(
                "Evaluation sweep checkpoint does not match the active runtime selection."
            )

    def _has_progress_callback(self) -> bool:
        """Return whether a live UI progress observer is attached."""

        with self._progress_callback_lock:
            return self._progress_callback is not None

    def _training_progress_reporter(self) -> ProgressCallback | None:
        """Return a callback that ends warmup reporting on first real progress."""

        if not self._has_progress_callback():
            return None
        return self._forward_training_progress

    def _forward_training_progress(
        self,
        total: float | None,
        progress: float | None,
        status: RLModelState,
        message: str,
    ) -> None:
        """Forward framework progress after atomically ending warmup status."""

        with self._training_warmup_lock:
            self._training_warmup_complete.set()
            self._update_progress(
                total=total,
                progress=progress,
                status=status,
                additional_status_message=message,
            )

    def _publish_training_warmup_heartbeat(self, run_id: str) -> None:
        """Publish periodic status-only updates until framework progress begins."""

        started_at = monotonic()
        interval = self._TRAINING_WARMUP_HEARTBEAT_SECONDS
        while not self._training_warmup_complete.wait(interval):
            snapshot = self._execution_coordinator.snapshot()
            if snapshot.run_id != run_id or snapshot.state not in {
                RLModelState.STARTING,
                RLModelState.RUNNING,
            }:
                return
            elapsed_seconds = max(1, int(monotonic() - started_at))
            with self._training_warmup_lock:
                if self._training_warmup_complete.is_set():
                    return
                self._update_progress(
                    total=-1,
                    progress=-1,
                    status=snapshot.state,
                    additional_status_message=(
                        f"Training warmup in progress ({elapsed_seconds} s elapsed)"
                    ),
                    run_id=run_id,
                )

    def _on_execution_transition(self, snapshot: ExecutionSnapshot) -> None:
        """Handle the execution transition event for this base model runtime."""
        if not self._execution_coordinator.is_current_run(snapshot.run_id):
            self._logger.debug(f"Ignored stale execution transition for run {snapshot.run_id}.")
            return

        messages = {
            RLModelState.STARTING: f"{self._running_mode.value} is starting.",
            RLModelState.RUNNING: f"{self._running_mode.value} started.",
            RLModelState.PAUSED: "Execution paused.",
            RLModelState.STOPPING: "Stop requested; waiting for cleanup.",
            RLModelState.STOPPED: f"{self._running_mode.value} stopped by user.",
            RLModelState.COMPLETED: "Model execution completed successfully.",
            RLModelState.FAILED: ("Model execution failed. Check the structured logs for details."),
        }
        message = messages.get(snapshot.state, snapshot.state.value)

        if snapshot.state == RLModelState.FAILED:
            failure = self._execution_coordinator.failure_for_run(snapshot.run_id)
            self._logger.critical("Model execution entered the failed state.")
            if failure is not None:
                self._logger.error(
                    f"{type(failure).__name__}: {failure}",
                    exc_info=(type(failure), failure, failure.__traceback__),
                )

        total: float | None = -1
        progress: float | None = -1
        if snapshot.state == RLModelState.STARTING:
            progress = 0
        elif snapshot.state == RLModelState.STOPPED:
            total = None
            progress = 0
        elif snapshot.state == RLModelState.COMPLETED:
            total = 100
            progress = 100

        self._update_progress(
            total=total,
            progress=progress,
            status=snapshot.state,
            additional_status_message=message,
            run_id=snapshot.run_id,
        )

    def _update_progress(
        self,
        total: float | None = -1,
        progress: float | None = -1,
        status: RLModelState = RLModelState.FAILED,
        additional_status_message: str = "",
        run_id: str | None = None,
    ) -> bool:
        """Update progress using the callback if available."""
        effective_run_id = (
            run_id
            or self._execution_coordinator.current_worker_run_id()
            or self._execution_coordinator.current_run_id
        )
        if run_id is not None and not self._execution_coordinator.is_current_run(run_id):
            self._logger.debug(f"Ignored stale progress update for run {run_id}.")
            return False

        self._last_execution_update = ModelExecutionUpdate(
            total=total,
            progress=progress,
            status=status,
            message=additional_status_message,
            run_id=effective_run_id,
        )
        self._logger.debug(
            f"Updating progress: total={total}, progress={progress}, status={status.value}"
        )
        with self._progress_callback_lock:
            progress_callback = self._progress_callback
        if progress_callback:
            try:
                progress_callback(
                    total,
                    progress,
                    status,
                    additional_status_message,
                )
            except Exception as exc:
                self._logger.debug(
                    f"Progress callback skipped due to UI shutdown/unavailability: {exc}"
                )
        return True

    def check_pause(self) -> None:
        """Block at a cooperative pause point until resume or stop."""
        self._execution_coordinator.check_pause()

    def check_stop(self) -> None:
        """Check if processing should stop and raise exception if needed."""
        self._execution_coordinator.check_stop()

    def pause(self) -> ExecutionSnapshot:
        """Pause the current operation."""
        snapshot = self._execution_coordinator.pause()
        self._logger.debug("Pause requested")
        return snapshot

    def resume(self) -> ExecutionSnapshot:
        """Resume the paused operation."""
        snapshot = self._execution_coordinator.resume()
        self._logger.debug("Resume requested")
        return snapshot

    def stop(self) -> ExecutionSnapshot:
        """Request cooperative stop without claiming that cleanup has finished."""
        handle = self._execution_coordinator.request_stop()
        self._logger.debug("Stop requested")
        return handle.snapshot()

    async def stop_and_wait(self, timeout_seconds: float = 30.0) -> ExecutionSnapshot:
        """Request stop and wait until the worker and cleanup have completed."""
        return await self._execution_coordinator.stop_and_wait(timeout_seconds)

    async def restart(self, stop_timeout_seconds: float = 30.0) -> ExecutionSnapshot:
        """Wait for the old worker before allocating a new run."""
        snapshot = self._execution_coordinator.snapshot()
        if snapshot.state in ACTIVE_EXECUTION_STATES:
            await self.stop_and_wait(stop_timeout_seconds)
        elif snapshot.run_id is not None:
            try:
                await self._execution_coordinator.wait_for_current(stop_timeout_seconds)
            except TimeoutError as error:
                raise ExecutionStopTimeoutError(
                    state=snapshot.state,
                    run_id=snapshot.run_id,
                    timeout_seconds=stop_timeout_seconds,
                ) from error
        return await self.start_operation()

    def is_alive(self) -> bool:
        """Return whether the coordinator still owns a live worker."""
        return self._execution_coordinator.is_alive()

    def join(self, timeout: float | None = None) -> None:
        """Synchronously join the current worker during process shutdown."""
        self._execution_coordinator.join(timeout)

    def register_execution_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a per-run resource cleanup hook."""
        self._resource_disposer.register_execution_cleanup(callback)

    def _cleanup_execution_resources(self) -> None:
        """Extension hook for framework-owned resources."""

    def _cleanup_execution_run(self, context: ExecutionContext) -> None:
        """Clean up the execution run owned by this base model runtime."""
        self._resource_disposer.cleanup_execution_run(context.run_id)

    def _report_execution_cleanup_failure(
        self,
        _error: BaseException,
    ) -> None:
        """Block new execution/administration until a recovery load succeeds."""

        reporter = self._degraded_state_reporter
        if reporter is not None:
            reporter(
                reload_required=True,
                storage_recovery_required=False,
            )

    def close_blocking(self) -> ExecutionSnapshot:
        """Dispose an idle runtime from a dedicated retirement worker.

        Active executions require the awaitable ``close_async`` path.  Runtime
        replacement holds the execution/load gate, so its retired instance is
        idle and can avoid nesting another executor inside the retirement
        worker's event loop.
        """

        return self._resource_disposer.close_blocking()

    async def close_async(self, timeout_seconds: float = 30.0) -> ExecutionSnapshot:
        """Idempotently dispose an initialized or executed model runtime.

        Marking disposal before awaiting stop closes the race in which a stale
        load candidate could otherwise start while the loader is discarding it.
        Framework adapters should register resources or override the focused
        cleanup hook instead of exposing process-global shutdown here.
        """
        return await self._resource_disposer.close_async(timeout_seconds)

    def _train(self) -> None:
        """Train the model with the provided arguments."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _evaluate(self) -> None:
        """Evaluate the model with the provided arguments."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _evaluate_library(
        self,
        execution: LibraryEvaluationExecution,
        context: ExecutionContext,
    ) -> EvaluationSweepResult:
        """Evaluate one explicit held-out library using framework-native inference."""

        raise NotImplementedError("Subclasses should implement library evaluation.")

    def _infer(self) -> None:
        """Run the selected checkpoint with deployment semantics."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _save(self, path: str) -> None:
        """Save the model to the specified path."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _load(self, path: str) -> None:
        """Load the model from the specified path."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_manifest(self) -> ModelManifest:
        """Get the model manifest associated with this model."""
        return self._model_manifest

    def get_tensorboard_writer(self) -> TensorboardWriter:
        """Get the TensorBoard writer for logging."""
        if self._running_mode is RLModelRunningModes.INFERENCE:
            return self._inference_writer
        with self._resource_disposer.resource_creation_guard():
            return self._tensorboard_writer_owner.require()
