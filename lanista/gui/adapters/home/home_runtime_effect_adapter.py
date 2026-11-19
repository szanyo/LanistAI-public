# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own Home-screen startup and runtime-activation Textual effects."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Protocol

from lanista.gui.services.home.home_configuration_flow_service import ConfigurationApplyPlan
from lanista.gui.services.home.home_startup_selection_flow_service import (
    HomeStartupSelectionFlowService,
    HomeStartupSelectionPlan,
)
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.evaluation_restart_contracts import (
    GuiEvaluationRestartContinuation,
)
from lanista.reinforcengine.rl_model.runtime.activation_service import RuntimeActivationRequest
from lanista.reinforcengine.rl_model.runtime.active_selection_service import (
    ActiveRuntimeSelectionRequest,
)
from lanista.reinforcengine.rl_model.runtime.events import RuntimeEventPublisher


class HomeRuntimeControlPanelPort(Protocol):
    """Control-panel effects needed by the Home runtime lifecycle."""

    def set_selected_framework_uuid(self, framework_uuid: str | None) -> None:
        """Set the selected framework uuid owned by this home runtime control panel port."""
        ...

    def refresh_runtime_context_from_model(self) -> None:
        """Refresh the runtime context from model owned by this home runtime
control panel port."""
        ...

    def complete_startup_bootstrap(self, force_open: bool = True) -> None:
        """Complete the startup bootstrap owned by this home runtime control
panel port."""
        ...

    def set_actions_locked(self, locked: bool) -> None:
        """Set the actions locked owned by this home runtime control panel port."""
        ...

    def start_progress_after_configuration_apply(self) -> None:
        """Start the progress after configuration apply workflow owned by this home runtime control
        panel port.
        """
        ...

    def prepare_gui_evaluation_continuation(
        self,
        continuation: GuiEvaluationRestartContinuation,
    ) -> None:
        """Prepare the gui evaluation continuation required by this home runtime control panel
        port.
        """
        ...


class _StartupFlowPort(Protocol):
    """Define the typed boundary for startup flow."""

    def build_startup_selection_plan(self) -> HomeStartupSelectionPlan:
        """Build the startup selection plan required by this startup flow port."""
        ...


class HomeRuntimeEffectAdapter:
    """Bind asynchronous runtime effects to one Home-screen lifetime.

    The application services remain the runtime state source.  This adapter
    owns only GUI observer tasks and a monotonically increasing screen-lifetime
    generation, so results from a previous mount cannot mutate a new view.
    """

    _STARTUP_DIALOG_DELAY_SECONDS = 0.15

    def __init__(
        self,
        *,
        services: ApplicationServices,
        startup_flow: HomeStartupSelectionFlowService,
        control_panel_provider: Callable[[], HomeRuntimeControlPanelPort],
        event_publisher_factory: Callable[[], RuntimeEventPublisher],
        model_loading_committer: Callable[[bool], None],
        ui_available: Callable[[], bool],
        schedule_ui: Callable[[float, Callable[[], None]], object],
        start_after_activation: Callable[[HomeRuntimeControlPanelPort], None],
        activation_failure_notifier: Callable[[str], None],
        logger: logging.Logger,
        restart_continuation_provider: Callable[
            [], GuiEvaluationRestartContinuation | None
        ] = lambda: None,
    ) -> None:
        """Initialize the home runtime effect adapter and its owned state."""
        self._services = services
        self._startup_flow: _StartupFlowPort = startup_flow
        self._control_panel_provider = control_panel_provider
        self._event_publisher_factory = event_publisher_factory
        self._model_loading_committer = model_loading_committer
        self._ui_available = ui_available
        self._schedule_ui = schedule_ui
        self._start_after_activation = start_after_activation
        self._activation_failure_notifier = activation_failure_notifier
        self._logger = logger
        self._restart_continuation_provider = restart_continuation_provider
        self._startup_task: asyncio.Task[None] | None = None
        self._activation_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._active = True
        self._startup_in_progress = False
        self._model_loading = False

    @property
    def startup_task(self) -> asyncio.Task[None] | None:
        """Return the observer task owned by the current screen mount."""

        return self._startup_task

    @property
    def activation_task(self) -> asyncio.Task[None] | None:
        """Return the current runtime-activation observer, when present."""

        return self._activation_task

    @property
    def startup_in_progress(self) -> bool:
        """Report whether startup selection still owns the Home bootstrap."""

        return self._startup_in_progress

    def mount(self) -> asyncio.Task[None]:
        """Create and retain the startup task for the current mount."""

        if not self._active:
            self._generation += 1
            self._active = True
        current = self._startup_task
        if current is not None and not current.done():
            return current
        generation = self._generation
        self._startup_in_progress = True
        task = asyncio.create_task(self._prepare_startup_selection(generation))
        self._startup_task = task
        return task

    async def close(self) -> None:
        """Invalidate late results, then cancel and join every GUI observer."""

        self._active = False
        self._generation += 1
        owned_tasks = tuple(
            task
            for task in (self._startup_task, self._activation_task)
            if task is not None
        )
        for task in owned_tasks:
            if not task.done():
                task.cancel()
        if owned_tasks:
            await asyncio.gather(*owned_tasks, return_exceptions=True)
        if self._startup_task in owned_tasks:
            self._startup_task = None
        if self._activation_task in owned_tasks:
            self._activation_task = None
        self._startup_in_progress = False
        self._set_model_loading(False)

    def schedule_activation(
        self,
        apply_plan: ConfigurationApplyPlan,
        control_panel: HomeRuntimeControlPanelPort,
        *,
        start_after_apply: bool,
    ) -> bool:
        """Start one activation observer and reject overlapping requests."""

        if not self._is_current(self._generation):
            return False
        current = self._activation_task
        if current is not None and not current.done():
            self._logger.warning(
                "Ignored overlapping runtime activation request owned by HomeScreen."
            )
            return False
        self._set_model_loading(True)
        control_panel.set_actions_locked(True)
        generation = self._generation
        task = asyncio.create_task(
            self._activate_runtime(
                generation,
                apply_plan,
                control_panel,
                start_after_apply=start_after_apply,
            )
        )
        self._activation_task = task
        return True

    async def _prepare_startup_selection(self, generation: int) -> None:
        """Prepare the startup selection required by this home runtime effect adapter."""
        panel: HomeRuntimeControlPanelPort | None = None
        bootstrap_completed = False
        try:
            # Yield once so an immediate unmount can invalidate startup before
            # it touches staged selection or Textual widgets.
            await asyncio.sleep(0)
            if not self._is_current(generation):
                return
            panel = self._control_panel_provider()
            plan = self._startup_flow.build_startup_selection_plan()
            self._logger.info(plan.log_message)
            continuation = self._restart_continuation_provider()
            if continuation is not None:
                await self._restore_gui_evaluation_continuation(
                    generation,
                    panel,
                    plan,
                    continuation,
                )
                bootstrap_completed = True
                return
            if plan.selected_manifest is not None:
                self._services.runtime.staging.stage_manifest(plan.selected_manifest)
            if plan.selected_framework_uuid is not None:
                panel.set_selected_framework_uuid(plan.selected_framework_uuid)
            if plan.refresh_control_panel:
                panel.refresh_runtime_context_from_model()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception("Home startup model selection failed unexpectedly.")
            if self._is_current(generation):
                self._activation_failure_notifier(
                    "Startup model selection failed. Check the logs for details."
                )
        finally:
            if generation == self._generation:
                self._startup_in_progress = False
            if (
                not bootstrap_completed
                and panel is not None
                and self._is_current(generation)
            ):
                self._schedule_guarded_bootstrap(panel, generation)
            if asyncio.current_task() is self._startup_task:
                self._startup_task = None

    async def _restore_gui_evaluation_continuation(
        self,
        generation: int,
        panel: HomeRuntimeControlPanelPort,
        plan: HomeStartupSelectionPlan,
        continuation: GuiEvaluationRestartContinuation,
    ) -> None:
        """Activate and validate the run selected before process replacement."""

        manifest = plan.selected_manifest
        if manifest is None:
            raise RuntimeError("The restarted evaluation model is unavailable.")
        configuration = manifest.get_configuration()
        if str(getattr(configuration, "model_uuid", "") or "") != continuation.model_uuid:
            raise RuntimeError("The restarted evaluation model identity changed.")
        self._services.runtime.staging.stage_manifest(manifest)
        panel.set_selected_framework_uuid(continuation.framework_uuid)
        result = await self._services.runtime.activation.execute(
            RuntimeActivationRequest(
                model_manifest=manifest,
                selected_framework_uuid=continuation.framework_uuid,
            ),
            event_publisher=self._event_publisher_factory(),
        )
        if not self._is_current(generation):
            return
        if not result.succeeded or result.value is None or not result.value.active_runtime_available:
            message = result.failure.message if result.failure is not None else "Runtime unavailable."
            raise RuntimeError(message)
        runtime = self._services.runtime.access.get_active_runtime()
        if runtime is None:
            raise RuntimeError("The restarted evaluation runtime is unavailable.")
        runtime.set_evaluation_scenario_index(continuation.scenario_index)
        selection = self._services.runtime.active_selection.set_selection(
            ActiveRuntimeSelectionRequest(
                session_name=continuation.session_name,
                checkpoint_name=continuation.checkpoint_name,
                persist=False,
                expected_model_uuid=continuation.model_uuid,
            )
        )
        if not selection.succeeded:
            message = selection.failure.message if selection.failure is not None else "Selection unavailable."
            raise RuntimeError(message)
        panel.refresh_runtime_context_from_model()
        panel.prepare_gui_evaluation_continuation(continuation)
        panel.complete_startup_bootstrap(force_open=False)

    def _schedule_guarded_bootstrap(
        self,
        panel: HomeRuntimeControlPanelPort,
        generation: int,
    ) -> None:
        """Schedule the guarded bootstrap owned by this home runtime effect adapter."""
        def complete() -> None:
            """Perform the complete step within the enclosing schedule guarded bootstrap workflow.
            """
            if self._is_current(generation):
                panel.complete_startup_bootstrap(force_open=True)

        self._schedule_ui(self._STARTUP_DIALOG_DELAY_SECONDS, complete)

    async def _activate_runtime(
        self,
        generation: int,
        apply_plan: ConfigurationApplyPlan,
        control_panel: HomeRuntimeControlPanelPort,
        *,
        start_after_apply: bool,
    ) -> None:
        """Activate the runtime owned by this home runtime effect adapter."""
        self._logger.info(
            "Import requested from Configure confirm: "
            f"'{apply_plan.import_model_name}'"
        )
        should_start = False
        try:
            result = await self._services.runtime.activation.execute(
                RuntimeActivationRequest(
                    model_manifest=apply_plan.staged_manifest,
                    selected_framework_uuid=apply_plan.selected_framework_uuid,
                ),
                event_publisher=self._event_publisher_factory(),
            )
            if self._is_current(generation):
                control_panel.refresh_runtime_context_from_model()
            should_start = bool(
                start_after_apply
                and result.succeeded
                and result.value is not None
                and result.value.active_runtime_available
            )
            if start_after_apply and not should_start:
                failure_code = (
                    result.failure.code
                    if result.failure is not None
                    else "empty_runtime"
                )
                self._logger.info(
                    "Automatic start canceled after model import result "
                    f"[{failure_code}; operation_id={result.operation_id}]."
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception("Runtime activation task failed unexpectedly.")
            if self._is_current(generation):
                self._activation_failure_notifier(
                    "The model runtime could not be activated. Check the logs for details."
                )
        finally:
            self._set_model_loading(False)
            if self._is_current(generation):
                control_panel.set_actions_locked(False)
            if asyncio.current_task() is self._activation_task:
                self._activation_task = None
        if should_start and self._is_current(generation):
            self._start_after_activation(control_panel)

    def _set_model_loading(self, loading: bool) -> None:
        """Set the model loading owned by this home runtime effect adapter."""
        if self._model_loading == loading:
            return
        self._model_loading = loading
        self._model_loading_committer(loading)

    def _is_current(self, generation: int) -> bool:
        """Return whether current applies to this home runtime effect adapter."""
        return bool(
            self._active
            and generation == self._generation
            and self._ui_available()
        )
