# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Execute ControlPanel start, resume, pause, and completion-observer effects."""

from __future__ import annotations

import asyncio
import logging
from asyncio import Task
from dataclasses import dataclass
from functools import partial
from typing import cast

from lanista.gui.adapters.control_panel.control_panel_attached_execution_handle import (
    ControlPanelAttachedExecutionHandle,
)
from lanista.gui.adapters.control_panel.control_panel_execution_metadata_effect_adapter import (
    ControlPanelExecutionMetadataEffectAdapter,
)
from lanista.gui.adapters.control_panel.control_panel_execution_presentation_adapter import (
    ControlPanelExecutionPresentationAdapter,
)
from lanista.gui.adapters.control_panel.control_panel_execution_projection import (
    TERMINAL_EXECUTION_STATES,
    ControlPanelExecutionProjection,
)
from lanista.gui.adapters.control_panel.control_panel_execution_task_owner import (
    ControlPanelExecutionTaskOwner,
)
from lanista.gui.adapters.control_panel.control_panel_execution_view_port import (
    ControlPanelExecutionViewPort,
)
from lanista.gui.adapters.control_panel.control_panel_library_execution_adapter import (
    prepare_library_execution,
)
from lanista.gui.contracts import normalize_ui_severity
from lanista.gui.services.async_collection_loader import run_blocking
from lanista.gui.services.control_panel.control_panel_execution_workflow import (
    ControlPanelExecutionWorkflowCoordinator,
)
from lanista.gui.services.control_panel.control_panel_execution_workflow_contracts import (
    ExecutionBeginCommand,
    ExecutionWorkflowHandle,
    ExecutionWorkflowLease,
)
from lanista.gui.services.control_panel.control_panel_operation_presenter import (
    ControlPanelOperationPresenter,
)
from lanista.gui.services.control_panel.control_panel_progress_coordinator import (
    ControlPanelProgressCoordinator,
)
from lanista.reinforcengine.application_errors import ApplicationError
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.evaluation.contracts import (
    EvaluationSweepProgress,
    LibraryEvaluationScope,
)
from lanista.reinforcengine.rl_model.evaluation.runtime_execution import (
    LibraryEvaluationExecution,
)
from lanista.reinforcengine.rl_model.evaluation_restart_contracts import (
    GuiEvaluationRestartContinuation,
)
from lanista.reinforcengine.rl_model.execution_contracts import (
    ACTIVE_EXECUTION_STATES,
    EvaluationPacing,
    EvaluationVisualization,
    ExecutionRejectedError,
    ExecutionSnapshot,
    RLModelState,
)
from lanista.reinforcengine.rl_model.model import BaseModelRuntime, RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.run_precheck_service import RunStartExecutionPlan
from lanista.reinforcengine.rl_model.runtime.status_service import RunModeExecutionPlan
from lanista.reinforcengine.utils.scenarios.scenario_selection import (
    resolve_scenario_index,
)


@dataclass(frozen=True, slots=True)
class _RuntimeStartPreparation:
    """Immutable runtime-only inputs safe to consume outside the UI thread."""

    plan: RunModeExecutionPlan
    evaluation_visualization: EvaluationVisualization
    evaluation_pacing: EvaluationPacing
    evaluation_scenario_index: int | None
    evaluation_library_scope: LibraryEvaluationScope | None = None
    selected_session: str | None = None
    selected_checkpoint: str | None = None


class ControlPanelExecutionCommandAdapter:
    """Run non-stop execution commands against coordinator-captured identity."""

    def __init__(
        self,
        *,
        view: ControlPanelExecutionViewPort,
        services: ApplicationServices,
        workflow: ControlPanelExecutionWorkflowCoordinator,
        projection: ControlPanelExecutionProjection,
        tasks: ControlPanelExecutionTaskOwner,
        presentation: ControlPanelExecutionPresentationAdapter,
        progress: ControlPanelProgressCoordinator,
        logger: logging.Logger,
        metadata: ControlPanelExecutionMetadataEffectAdapter,
    ) -> None:
        """Initialize the control panel execution command adapter and its owned state."""
        self._view = view
        self._services = services
        self._workflow = workflow
        self._projection = projection
        self._tasks = tasks
        self._presentation = presentation
        self._progress = progress
        self._logger = logger
        self._metadata = metadata
        self._operation_presenter = ControlPanelOperationPresenter()
        self._active_gui_evaluation_restart: GuiEvaluationRestartContinuation | None = None

    def start(self) -> None:
        """Start the workflow owned by this control panel execution command adapter."""
        self._view.refresh_execution_runtime()
        if self._view.native_gui_restart_required():
            self._view.notify_execution(
                "Restart LanistAI before starting another run.",
                title="Restart required",
                severity="warning",
            )
            self._view.refresh_execution_buttons()
            return
        snapshot = self._projection.synchronize()
        if (
            snapshot.worker_alive and snapshot.state is not RLModelState.PAUSED
        ) or snapshot.state in {
            RLModelState.STARTING,
            RLModelState.RUNNING,
            RLModelState.STOPPING,
        }:
            self._view.notify_execution(
                f"Execution is already {snapshot.state.value}.",
                title="Start blocked",
                severity="warning",
            )
            self._view.refresh_execution_buttons()
            return
        selection = self._services.runtime.context.get_current_runtime_selection_state()
        mode = self._view.selected_execution_mode()
        context = self._operation_presenter.build_start_context(
            current_mode=mode,
            selected_session=self._view.selected_execution_session(),
            selected_checkpoint=self._view.selected_execution_checkpoint(),
            runtime_session=selection.session_name,
            runtime_checkpoint=selection.checkpoint_name,
        )
        if context.blocked_notification is not None:
            self._view.notify_execution(
                context.blocked_notification.message,
                title=context.blocked_notification.title,
                severity=context.blocked_notification.severity,
            )
            return
        plan = self._services.runtime.run_precheck.build_run_start_execution_plan(
            mode=mode,
            selected_session=context.selected_session,
            selected_checkpoint=context.selected_checkpoint,
            has_previous_models=self._services.catalog.models.has_available_models(),
            panel_state=snapshot.state,
        )
        if plan.action == "blocked":
            self._apply_blocked_start(plan)
            return
        if plan.action == "resume":
            self.resume()
            return
        self._view.set_execution_selection(
            context.selected_session,
            context.selected_checkpoint,
        )
        self._start_new_operation()

    def reconcile_attached_runtime(
        self,
        runtime: BaseModelRuntime,
        snapshot: ExecutionSnapshot,
        *,
        project_detached_terminal: bool,
    ) -> None:
        """Restore GUI observation after remount without owning the worker."""

        if project_detached_terminal and snapshot.state in TERMINAL_EXECUTION_STATES:
            self._projection.apply_terminal_effects_once(snapshot, "")
            self._view.refresh_execution_buttons()
            self._view.publish_execution_context()
            return
        if snapshot.state not in ACTIVE_EXECUTION_STATES or self._tasks.operation_is_running():
            return
        handle = ControlPanelAttachedExecutionHandle(runtime, snapshot.run_id or "")
        selection = self._services.runtime.context.get_current_runtime_selection_state()
        lease = self._workflow.adopt_attached_handle(
            runtime,
            handle,
            snapshot,
            selected_session=(self._view.selected_execution_session() or selection.session_name),
        )
        if lease is None:
            return
        try:
            self._progress.bind(runtime)
        except Exception:
            self._logger.exception(
                "Execution progress could not be rebound after ControlPanel remount."
            )
            self._view.notify_execution(
                "Execution is still running, but live progress could not be restored.",
                title="Progress unavailable",
                severity="warning",
            )
        observer = asyncio.create_task(self.observe_completion(lease))
        self._tasks.track_operation(observer)

    def _apply_blocked_start(self, plan: RunStartExecutionPlan) -> None:
        """Apply the blocked start transition owned by this control panel execution command
        adapter.
        """
        if plan.require_load_model:
            self._view.request_model_load()
        if plan.message:
            self._view.notify_execution(
                plan.message,
                title=plan.title,
                severity=normalize_ui_severity(plan.severity),
            )
        if plan.require_apply_configuration:
            self._view.request_configuration_apply()
            self._view.refresh_execution_buttons()
            return
        if plan.require_dialog:
            self._view.request_configuration_dialog()
        self._view.refresh_execution_buttons()

    def resume(self) -> None:
        """Resume the operation owned by this control panel execution command adapter."""
        runtime = self._projection.current_runtime()
        if runtime is None:
            self._reject("Resume", "No active model runtime is available.")
            return
        command = self._workflow.capture_command(runtime)
        if command is None:
            self._reject("Resume", "No owned execution run is available.")
            return
        try:
            snapshot = runtime.resume()
        except ExecutionRejectedError as error:
            self._reject("Resume", str(error))
            return
        if not self._projection.apply_transition(
            self._workflow.accept_command_result(command, snapshot)
        ):
            return
        self._view.set_execution_status("running", "Execution resumed")
        self._view.refresh_execution_buttons()
        self._metadata.persist_session_status(
            self._workflow.context.originating_session or self._view.selected_execution_session(),
            RLModelState.RUNNING,
            action_summary="Execution resumed",
        )
        self._presentation.notify_feedback("resumed")

    def pause(self) -> None:
        """Handle the pause operation owned by this control panel execution command adapter."""
        runtime = self._projection.current_runtime()
        if runtime is None:
            self._reject("Pause", "No active model runtime is available.")
            return
        command = self._workflow.capture_command(runtime)
        if command is None:
            self._reject("Pause", "No owned execution run is available.")
            return
        try:
            snapshot = runtime.pause()
        except ExecutionRejectedError as error:
            self._reject("Pause", str(error))
            return
        if not self._projection.apply_transition(
            self._workflow.accept_command_result(command, snapshot)
        ):
            return
        self._view.set_execution_status("paused", "Execution paused")
        self._view.refresh_execution_buttons()
        self._metadata.persist_session_status(
            self._workflow.context.originating_session or self._view.selected_execution_session(),
            RLModelState.PAUSED,
            action_summary="Execution paused",
        )
        self._presentation.notify_feedback("paused")

    def _start_new_operation(self) -> None:
        """Start the new operation workflow owned by this control panel execution command adapter."""
        if self._tasks.begin_is_running() or self._tasks.operation_is_running():
            return
        runtime = self._projection.current_runtime()
        if runtime is None:
            self._reject("Start", "No active model runtime is available.")
            return
        try:
            self._prepare_gui_evaluation_restart(runtime)
        except (ApplicationError, ValueError) as error:
            self._reject("Start", str(error))
            return
        command = self._workflow.begin(
            runtime,
            selected_session=self._view.selected_execution_session(),
        )
        if command is None:
            self._reject(
                "Start",
                "The active runtime changed before execution could begin.",
            )
            return
        pending = self._workflow.project_pending_begin(command)
        if not self._projection.apply_transition(pending):
            self._reconcile_start_exception(
                command,
                error=RuntimeError("The pending start could not be projected."),
            )
            self._reject(
                "Start",
                "The execution start reservation is no longer current.",
            )
            return

        self._view.set_execution_status(
            RLModelState.STARTING.value,
            f"Preparing {self._view.selected_execution_mode().value.lower()} run",
        )
        self._view.refresh_execution_buttons()
        try:
            preparation = self._capture_runtime_start_preparation()
        except (ApplicationError, ValueError) as error:
            self._reconcile_start_exception(command, error=error)
            self._reject("Start", str(error))
            return
        except Exception as error:
            self._logger.exception("Execution preparation planning failed.")
            self._reconcile_start_exception(command, error=error)
            self._view.notify_execution(
                "Execution could not be prepared. Check the logs for details.",
                title="Start failed",
                severity="error",
            )
            self._view.refresh_execution_buttons()
            return
        try:
            self._progress.bind(runtime)
        except Exception as error:
            self._logger.exception("Execution progress could not be bound before start.")
            self._reconcile_start_exception(command, error=error)
            self._reject("Start", "Execution progress is unavailable.")
            return

        begin_task = asyncio.create_task(
            self._begin_runtime_operation_async(
                runtime,
                command,
                preparation,
            )
        )
        self._tasks.track_begin(begin_task)

    def _prepare_gui_evaluation_restart(
        self,
        runtime: BaseModelRuntime,
    ) -> None:
        """Prepare the gui evaluation restart required by this control panel execution command
        adapter.
        """
        mode = self._view.selected_execution_mode()
        visualization = self._view.selected_evaluation_visualization()
        if (
            mode is not RLModelRunningModes.EVALUATION
            or visualization != "gui"
            or self._view.selected_evaluation_library_scope() is not None
        ):
            self._active_gui_evaluation_restart = None
            self._view.set_post_run_continuation(None)
            return

        continuation = self._view.evaluation_restart_continuation()
        configuration = runtime.get_manifest().get_configuration()
        selected_scenario_index = self._view.selected_evaluation_scenario_index()
        scenario_index = (
            selected_scenario_index
            if selected_scenario_index is not None
            else resolve_scenario_index(
                configuration,
                purpose="evaluation",
                default_index=0,
            )
        )
        if continuation is None:
            continuation = GuiEvaluationRestartContinuation(
                model_uuid=str(getattr(configuration, "model_uuid", "") or ""),
                framework_uuid=str(getattr(runtime, "framework_uuid", "") or "").lower(),
                session_name=str(self._view.selected_execution_session() or ""),
                checkpoint_name=self._view.selected_execution_checkpoint(),
                pacing=self._view.selected_evaluation_pacing(),
                scenario_index=scenario_index,
            )
        self._validate_gui_evaluation_continuation(runtime, continuation)
        self._active_gui_evaluation_restart = continuation
        self._view.set_post_run_continuation(continuation)

    def _validate_gui_evaluation_continuation(
        self,
        runtime: BaseModelRuntime,
        continuation: GuiEvaluationRestartContinuation,
    ) -> None:
        """Validate the gui evaluation continuation required by this control panel execution
        command adapter.
        """
        configuration = runtime.get_manifest().get_configuration()
        selected_scenario_index = self._view.selected_evaluation_scenario_index()
        current = (
            str(getattr(configuration, "model_uuid", "") or ""),
            str(getattr(runtime, "framework_uuid", "") or "").lower(),
            str(self._view.selected_execution_session() or ""),
            self._view.selected_execution_checkpoint(),
            (
                selected_scenario_index
                if selected_scenario_index is not None
                else resolve_scenario_index(
                    configuration,
                    purpose="evaluation",
                    default_index=0,
                )
            ),
        )
        expected = (
            continuation.model_uuid,
            continuation.framework_uuid,
            continuation.session_name,
            continuation.checkpoint_name,
            continuation.scenario_index,
        )
        if current != expected:
            raise ApplicationError(
                "The restarted GUI evaluation no longer matches the active runtime selection."
            )

    def _begin_runtime_operation(
        self,
        runtime: BaseModelRuntime,
        preparation: _RuntimeStartPreparation,
    ) -> ExecutionWorkflowHandle:
        """Begin one operation from a non-UI thread."""

        scope = getattr(preparation, "evaluation_library_scope", None)
        if scope is not None:
            execution = self._build_library_execution(runtime, preparation)
            return runtime.begin_evaluation_library(execution)
        return runtime.begin_operation(
            prepare=partial(
                self._prepare_runtime_for_start,
                runtime,
                preparation,
            )
        )

    def _build_library_execution(
        self, runtime: BaseModelRuntime, preparation: _RuntimeStartPreparation
    ) -> LibraryEvaluationExecution:
        scope = preparation.evaluation_library_scope
        if scope is None:
            raise ValueError("No evaluation library was selected.")
        return prepare_library_execution(
            runtime=runtime,
            scope=scope,
            library_service=self._services.evaluation_libraries,
            checkpoint_catalog=self._services.catalog.checkpoints,
            session_name=preparation.selected_session,
            checkpoint_name=preparation.selected_checkpoint,
            pacing=preparation.evaluation_pacing,
            progress_callback=self._on_library_progress,
        )

    def _on_library_progress(self, progress: EvaluationSweepProgress) -> None:
        attempted = (
            progress.completed_scenario_count
            + progress.failed_scenario_count
            + progress.cancelled_scenario_count
        )
        groups = "; ".join(
            f"{group.name} [completed {group.completed_scenario_count}, "
            f"failed {group.failed_scenario_count}, "
            f"cancelled {group.cancelled_scenario_count}, "
            f"remaining {group.remaining_scenario_count}]"
            for group in progress.groups
        )
        message = (
            f"Library: completed {progress.completed_scenario_count}, "
            f"failed {progress.failed_scenario_count}, "
            f"cancelled {progress.cancelled_scenario_count}, "
            f"remaining {progress.remaining_scenario_count} "
            f"({progress.expected_scenario_count} total). Groups: {groups}"
        )
        self._view.dispatch_execution_ui(
            partial(
                self._view.set_execution_status,
                RLModelState.RUNNING.value,
                message,
                float(attempted),
                float(progress.expected_scenario_count),
            )
        )

    async def _begin_runtime_operation_async(
        self,
        runtime: BaseModelRuntime,
        command: ExecutionBeginCommand,
        preparation: _RuntimeStartPreparation,
    ) -> None:
        """Await blocking preparation while keeping Textual responsive."""

        try:
            handle = await run_blocking(
                partial(
                    self._begin_runtime_operation,
                    runtime,
                    preparation,
                )
            )
        except (ApplicationError, ExecutionRejectedError, ValueError) as error:
            if (
                not self._reconcile_start_exception(command, error=error)
                and self._view.execution_effects_available()
            ):
                self._reject("Start", str(error))
            self._active_gui_evaluation_restart = None
            return
        except Exception as error:
            self._logger.exception("Execution preparation failed before worker start.")
            self._reconcile_start_exception(command, error=error)
            if self._view.execution_effects_available():
                self._view.notify_execution(
                    "Execution could not be prepared. Check the logs for details.",
                    title="Start failed",
                    severity="error",
                )
                self._view.refresh_execution_buttons()
            self._active_gui_evaluation_restart = None
            return
        finally:
            current = asyncio.current_task()
            self._tasks.release_begin(current if isinstance(current, Task) else None)

        if self._adopt_started_operation(command, handle) and preparation.plan.start_tensorboard:
            self._view.start_tensorboard_for_execution()

    def _adopt_started_operation(
        self,
        command: ExecutionBeginCommand,
        handle: ExecutionWorkflowHandle,
    ) -> bool:
        """Adopt the started operation owned by this control panel execution
        command adapter."""
        snapshot = handle.snapshot()
        adoption = self._workflow.adopt(command, handle, snapshot)
        if adoption is None:
            return False
        runtime = cast(BaseModelRuntime, command.runtime)
        if not self._projection.apply_transition(adoption.transition):
            return False
        if not adoption.contract_matches:
            self._logger.error(
                "Execution handle returned a snapshot for a different run "
                "[handle=%s; snapshot=%s].",
                handle.run_id,
                snapshot.run_id,
            )
            self._stop_contract_violating_start(runtime, snapshot)
            self._view.refresh_execution_buttons()
            self._view.notify_execution(
                "Execution returned inconsistent run ownership and was stopped.",
                title="Start failed",
                severity="error",
            )
            return False
        lease = adoption.lease
        if lease is None:
            raise RuntimeError("A contract-matching execution adoption requires a lease.")
        observer = asyncio.create_task(self.observe_completion(lease))
        self._tasks.track_operation(observer)
        if snapshot.state in TERMINAL_EXECUTION_STATES:
            self._projection.apply_terminal_progress(
                runtime,
                snapshot,
                self._execution_update_message(runtime, snapshot),
                expected_task=observer,
                lease=lease,
            )
            return False
        self._view.refresh_execution_buttons()
        self._view.publish_execution_context()
        self._metadata.persist_session_status(
            self._workflow.context.originating_session,
            snapshot.state,
            action_summary="Execution started",
        )
        self._view.set_execution_status(
            snapshot.state.value,
            f"Preparing {self._view.selected_execution_mode().value.lower()} run",
        )
        self._presentation.notify_feedback("started")
        return True

    def _reconcile_start_exception(
        self,
        command: ExecutionBeginCommand,
        *,
        error: Exception,
    ) -> bool:
        """Return the reconcile start exception value owned by this control panel execution command
        adapter.
        """
        runtime = cast(BaseModelRuntime, command.runtime)
        snapshot = runtime.get_execution_snapshot()
        reconciliation = self._workflow.reconcile_begin_failure(command, snapshot)
        if reconciliation is None or not reconciliation.adopted_new_run:
            return False
        self._projection.apply_transition(reconciliation.transition)
        if snapshot.state in TERMINAL_EXECUTION_STATES:
            self._projection.apply_terminal_progress(runtime, snapshot, str(error))
        else:
            self._stop_contract_violating_start(runtime, snapshot)
            self._view.refresh_execution_buttons()
        return True

    def _stop_contract_violating_start(
        self,
        runtime: BaseModelRuntime,
        snapshot: ExecutionSnapshot,
    ) -> None:
        """Stop the contract violating start workflow owned by this control panel execution command
        adapter.
        """
        current = runtime.get_execution_snapshot()
        if current.run_id != snapshot.run_id or current.state not in ACTIVE_EXECUTION_STATES:
            return
        try:
            stopping = runtime.stop()
        except Exception:
            self._logger.exception(
                "Could not stop contract-violating execution run %s.",
                snapshot.run_id,
            )
            return
        self._projection.apply_snapshot(runtime, stopping)

    @staticmethod
    def _execution_update_message(
        runtime: BaseModelRuntime,
        snapshot: ExecutionSnapshot,
    ) -> str:
        """Return the execution update message value owned by this control panel execution command
        adapter.
        """
        update = runtime.get_last_execution_update()
        if update is None or update.run_id != snapshot.run_id:
            return ""
        return update.message

    async def observe_completion(
        self,
        lease: ExecutionWorkflowLease,
    ) -> ExecutionSnapshot:
        """Run the observe completion responsibility for this control panel execution command
        adapter asynchronously.
        """
        runtime = cast(BaseModelRuntime, lease.runtime)
        snapshot = await lease.handle.wait()
        current = asyncio.current_task()
        expected = current if isinstance(current, Task) else None
        message = self._execution_update_message(runtime, snapshot)
        if snapshot.state in TERMINAL_EXECUTION_STATES:
            self._projection.apply_terminal_progress(
                runtime,
                snapshot,
                message,
                expected_task=expected,
                lease=lease,
            )
        elif self._projection.apply_snapshot(
            runtime,
            snapshot,
            expected_task=expected,
            lease=lease,
        ):
            self._view.refresh_execution_buttons()
        self._handle_gui_evaluation_terminal(snapshot)
        return snapshot

    def _handle_gui_evaluation_terminal(self, snapshot: ExecutionSnapshot) -> None:
        """Publish a completed GUI run for an explicit user restart action."""
        continuation = self._active_gui_evaluation_restart
        if continuation is None or snapshot.state not in TERMINAL_EXECUTION_STATES:
            return
        self._active_gui_evaluation_restart = None
        if not self._view.native_gui_restart_required():
            return
        self._view.set_post_run_continuation(continuation)
        self._view.refresh_execution_buttons()

    def configure_running_mode(self) -> None:
        """Configure the running mode owned by this control panel execution command adapter."""
        runtime = self._projection.current_runtime()
        if runtime is None:
            raise RuntimeError("No active runtime is available during run preparation.")
        preparation = self._capture_runtime_start_preparation()
        self._progress.bind(runtime)
        self._prepare_runtime_for_start(runtime, preparation)
        if preparation.plan.start_tensorboard:
            self._view.start_tensorboard_for_execution()

    def _capture_runtime_start_preparation(self) -> _RuntimeStartPreparation:
        """Capture UI-owned choices before crossing into the preparation thread."""

        mode = self._view.selected_execution_mode()
        scope = self._view.selected_evaluation_library_scope()
        if scope is not None and mode is not RLModelRunningModes.EVALUATION:
            raise ValueError("A library can only be run in Evaluation mode.")
        if scope is not None and self._view.selected_evaluation_scenario_index() is not None:
            raise ValueError("A library and one scenario cannot be selected together.")
        return _RuntimeStartPreparation(
            plan=self._services.runtime.status.build_run_mode_execution_plan(mode),
            evaluation_visualization=(
                "headless" if scope is not None
                else self._view.selected_evaluation_visualization()
            ),
            evaluation_pacing=(
                "maximum-throughput" if scope is not None
                else self._view.selected_evaluation_pacing()
            ),
            evaluation_scenario_index=(
                None if scope is not None
                else self._view.selected_evaluation_scenario_index()
            ),
            evaluation_library_scope=scope,
            selected_session=self._view.selected_execution_session(),
            selected_checkpoint=self._view.selected_execution_checkpoint(),
        )

    @staticmethod
    def _prepare_runtime_for_start(
        runtime: BaseModelRuntime,
        preparation: _RuntimeStartPreparation,
    ) -> None:
        """Apply runtime-only preparation without touching Textual state."""

        plan = preparation.plan
        runtime.set_running_mode(plan.mode)
        if plan.mode is RLModelRunningModes.EVALUATION:
            runtime.set_evaluation_profile(
                preparation.evaluation_visualization,
                preparation.evaluation_pacing,
            )
            runtime.set_evaluation_scenario_index(
                preparation.evaluation_scenario_index,
            )
        if plan.refresh_tensorboard_writer:
            runtime.refresh_tensorboard_writer()

    def _reject(self, action: str, message: str) -> None:
        """Reject the operation owned by this control panel execution command adapter."""
        self._projection.synchronize()
        self._view.refresh_execution_buttons()
        self._presentation.notify_command_rejection(action, message)
