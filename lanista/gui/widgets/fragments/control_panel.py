# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the control panel interface component."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from threading import get_ident
from time import monotonic
from typing import cast

from textual._context import NoActiveAppError
from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Button, ProgressBar, Select, Static

from lanista.application import LanistAIApplication
from lanista.foundation.configuration import AppConfiguration
from lanista.gui.adapters.control_panel.control_panel_configuration_dialog_adapter import (
    ConfigurationScreenActionsPort,
)
from lanista.gui.adapters.control_panel.control_panel_configuration_effect_adapter import (
    ControlPanelConfigurationEffectAdapter,
)
from lanista.gui.adapters.control_panel.control_panel_execution_effect_adapter import (
    ControlPanelExecutionEffectAdapter,
)
from lanista.gui.adapters.control_panel.control_panel_inspection_effect_adapter import (
    ControlPanelInspectionEffectAdapter,
)
from lanista.gui.contracts import UiNotification, UiSeverity
from lanista.gui.predefs.secure_yes_no_dialog_definitions import secure_stop_training_dialog
from lanista.gui.services.async_collection_loader import run_blocking
from lanista.gui.services.configuration.control_panel_configuration_contracts import (
    ConfigurationDialogResult,
    ConfigurationDialogSelection,
    EvaluationTarget,
)
from lanista.gui.services.configuration.control_panel_framework_selection_service import (
    ControlPanelFrameworkSelectionService,
)
from lanista.gui.services.control_panel.control_panel_execution_workflow import (
    ControlPanelExecutionWorkflowCoordinator,
)
from lanista.gui.services.control_panel.control_panel_operation_presenter import (
    ControlPanelButtonPresentation,
    ControlPanelButtonsPresentation,
    ControlPanelOperationPresenter,
)
from lanista.gui.services.control_panel.control_panel_progress_coordinator import (
    ControlPanelProgressCoordinator,
    ControlPanelProgressEvent,
)
from lanista.gui.services.control_panel.execution_elapsed_time import (
    ExecutionElapsedTime,
)
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.widgets.action_rows import action_row
from lanista.gui.widgets.dialogs.configuration_dialog import RunConfigurationDialog
from lanista.inspection_lifecycle import InspectionStatus
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.evaluation.contracts import LibraryEvaluationScope
from lanista.reinforcengine.rl_model.evaluation_restart_contracts import (
    GuiEvaluationRestartContinuation,
)
from lanista.reinforcengine.rl_model.execution_contracts import (
    ACTIVE_EXECUTION_STATES,
    EvaluationPacing,
    EvaluationVisualization,
    ExecutionSnapshot,
    RLModelState,
)
from lanista.reinforcengine.rl_model.inspection_target_service import InspectionTarget
from lanista.reinforcengine.rl_model.model import BaseModelRuntime, RLModelRunningModes

APP_CONFIGURATION = AppConfiguration()
LOGGER = logging.getLogger(__name__)


class ControlPanel(DirectionalFocusMixin, Vertical):
    """Present control in the Textual interface."""

    _RL_FRAMEWORKS = getattr(APP_CONFIGURATION, "rl_frameworks", {})
    _APPLICATION = LanistAIApplication()
    _NOTIFY_TIMEOUT_SECONDS = 30.0
    _STOP_TIMEOUT_SECONDS = 30.0

    panel_state: reactive[RLModelState] = reactive(RLModelState.IDLE)
    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "start": {"right": "resume"},
        "resume": {"left": "start", "right": "pause"},
        "pause": {"left": "resume", "right": "stop"},
        "stop": {"left": "pause"},
        "run_again": {"right": "change_configuration"},
        "change_configuration": {"left": "run_again"},
    }

    @dataclass
    class ExecutionSnapshotChanged(Message):
        """Carry the execution snapshot changed interface message."""

        snapshot: ExecutionSnapshot
        commands_locked: bool

    @dataclass
    class RunningModeChanged(Message):
        """Carry the running mode changed interface message."""

        mode: RLModelRunningModes

    @dataclass
    class ModeUpdateRequested(Message):
        """Carry the mode update requested interface message."""

        mode: RLModelRunningModes

    @dataclass
    class RuntimeContextChanged(Message):
        """Carry the runtime context changed interface message."""

        mode: RLModelRunningModes
        session_name: str | None
        parent_session_name: str | None
        checkpoint_name: str | None
        tensorboard_status: str
        session_created_at_raw: str | None

    def __init__(
        self,
        *,
        application_services: ApplicationServices | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        markup: bool = True,
    ) -> None:
        """Initialize the control panel and its owned state."""
        super().__init__(
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
            markup=markup,
        )
        self._services = application_services or self._APPLICATION.get_application_services()
        self.border_title = "Control Panel"
        self._unmounting = False
        self._selected_session: str | None = None
        self._selected_checkpoint: str | None = None
        self._evaluation_visualization: EvaluationVisualization = "gui"
        self._evaluation_pacing: EvaluationPacing = "maximum-throughput"
        self._evaluation_scenario_index: int | None = None
        self._evaluation_library_scope: LibraryEvaluationScope | None = None
        self._evaluation_model_uuid: str | None = None
        self._evaluation_restart_continuation: GuiEvaluationRestartContinuation | None = None
        self._current_mode: RLModelRunningModes = RLModelRunningModes.TRAINING
        self._startup_dialog_shown_once: bool = False
        self._startup_bootstrap_ready: bool = False
        self._actions_locked: bool = False
        self._framework_selection_service = ControlPanelFrameworkSelectionService()
        self._execution_workflow = ControlPanelExecutionWorkflowCoordinator()
        self._operation_presenter = ControlPanelOperationPresenter()
        self._progress_coordinator = ControlPanelProgressCoordinator(
            dispatcher=self,
            sink=self,
            logger=LOGGER,
        )
        self._execution_elapsed = ExecutionElapsedTime()
        self._elapsed_refresh_timer: Timer | None = None
        self._execution_effects = ControlPanelExecutionEffectAdapter(
            view=self,
            services=self._services,
            workflow=self._execution_workflow,
            progress=self._progress_coordinator,
            logger=LOGGER,
            stop_timeout_seconds=self._STOP_TIMEOUT_SECONDS,
        )
        self._framework_options = self._framework_selection_service.build_framework_options(
            self._RL_FRAMEWORKS
        )
        self._selected_framework_uuid: str = (
            self._framework_options[0][1] if self._framework_options else "ray"
        )
        self._configuration_effects = ControlPanelConfigurationEffectAdapter(
            view=self,
            services=self._services,
            framework_options=self._framework_options,
        )
        self._inspection_effects = ControlPanelInspectionEffectAdapter(
            view=self,
            services=self._services,
            stop_tensorboard=self._APPLICATION.stop_tensorboard,
            restart_tensorboard_for_mode=self._APPLICATION.restart_tensorboard_for_mode,
            active_logdir=self._APPLICATION.get_active_tensorboard_logdir,
            start_inspection=self._start_inspection_in_background,
            inspection_status=getattr(
                self._APPLICATION,
                "get_inspection_status",
                None,
            ),
            open_native=getattr(self._APPLICATION, "open_native_inspection", None),
            open_analysis=getattr(
                self._APPLICATION,
                "open_analysis_inspection",
                None,
            ),
        )

    def on_key(self, event: Key) -> None:
        """Handle the key event for this control panel."""
        direction_map = {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
        }
        direction = direction_map.get(event.key)
        if direction is None:
            return

        focused = self.app.focused
        focused_id = focused.id if focused is not None else None
        if focused_id not in self._NAVIGATION_GRAPH:
            return

        if self._move_focus_in_direction(direction):
            event.stop()

    def get_selected_framework_uuid(self) -> str:
        """Return the selected framework uuid exposed by this control panel."""
        return self._selected_framework_uuid

    def set_selected_framework_uuid(self, framework_uuid: str | None) -> None:
        """Set the selected framework uuid owned by this control panel."""
        selected_framework_uuid = (
            self._framework_selection_service.normalize_requested_framework_uuid(
                requested_framework_uuid=framework_uuid,
                framework_options=self._framework_options,
                fallback_framework_uuid=self._selected_framework_uuid,
            )
        )
        if selected_framework_uuid != self._selected_framework_uuid:
            self._evaluation_scenario_index = None
        self._selected_framework_uuid = selected_framework_uuid

    def _resolved_selected_framework_uuid(self) -> str:
        """Return the resolved selected framework uuid value owned by this control panel."""
        model_framework_uuid = self._current_framework_uuid()
        return self._framework_selection_service.resolve_selected_framework_uuid(
            model_framework_uuid=model_framework_uuid,
            selected_framework_uuid=self._selected_framework_uuid,
            framework_options=self._framework_options,
        )

    def _current_framework_uuid(self) -> str | None:
        """Return the current framework uuid value owned by this control panel."""
        runtime_context = self._services.runtime.context
        return self._services.runtime.selection.resolve_current_model_framework_uuid(
            configuration=runtime_context.get_active_configuration(),
            current_runtime_framework_uuid=(runtime_context.get_current_framework_uuid()),
        )

    def get_selected_session_name(self) -> str | None:
        """Return the selected session name exposed by this control panel."""
        return self._selected_session

    def get_selected_checkpoint_name(self) -> str | None:
        """Return the selected checkpoint name exposed by this control panel."""
        return self._selected_checkpoint

    def get_selected_mode(self) -> RLModelRunningModes:
        """Return the selected mode exposed by this control panel."""
        return self._current_mode

    def _current_runtime(self) -> BaseModelRuntime | None:
        """Return the current runtime value owned by this control panel."""
        return self._execution_effects.current_runtime()

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this control panel."""
        with Static(id="rl_model_mode_tabs_content"):
            with Horizontal(id="progress_container"):
                yield ProgressBar(id="progress", show_eta=True)
                yield Static("Elapsed: —", id="progress_elapsed")
            with Center(id="rl_model_status_container"), Vertical(id="rl_status_stack"):
                yield Static("Activity: idle", id="rl_model_status")
                yield Static("Step: -", id="rl_model_substatus")
            with Horizontal(id="button_container") as button_container:
                button_container.display = False
                yield Button("Start", id="start", variant="success")
                yield Button("Resume", id="resume", variant="success")
                yield Button("Pause", id="pause", variant="primary")
                yield Button("Stop", id="stop", variant="error")
                yield Button("Finish inspection", id="finish_inspection", variant="warning")
                yield Button("Run again", id="run_again", variant="success")
                yield Button(
                    "Change configuration",
                    id="change_configuration",
                    variant="primary",
                )
            with Vertical(id="inspection_server_status") as inspection_status:
                inspection_status.display = False
                yield Static("Native TensorBoard: stopped", id="inspection_native_status")
                yield Static("Analysis Dashboard: stopped", id="inspection_analysis_status")
            yield action_row(
                Button("Open TensorBoard", id="open_native_inspection", variant="primary"),
                Button("Open Analysis", id="open_analysis_inspection", variant="primary"),
                id="inspection_viewer_container",
                displayed=False,
                classes="action-row--compact",
            )

    def on_mount(self) -> None:
        """Handle the mount event for this control panel."""
        self._unmounting = False
        self._execution_effects.activate()
        self._inspection_effects.activate()
        self._progress_coordinator.activate(
            dispatcher=self,
            sink=self,
        )
        if self._elapsed_refresh_timer is None:
            self._elapsed_refresh_timer = self.set_interval(
                1.0,
                self._refresh_execution_elapsed,
            )
        else:
            self._elapsed_refresh_timer.resume()
        self._update_buttons()
        self.refresh_runtime_context_from_model()

    def _start_inspection_in_background(self, target: InspectionTarget) -> None:
        """Keep execution-triggered inspection startup off the UI thread."""

        async def _start() -> None:
            try:
                status = await run_blocking(
                    partial(
                        self._APPLICATION.start_inspection,
                        target,
                    )
                )
            except Exception:
                self._logger.exception("Inspection startup failed.")
                status = self._current_inspection_server_status()
            if not self.inspection_effects_available():
                return
            self.set_inspection_server_status(status)
            self._publish_runtime_context()

        self.run_worker(
            _start(),
            name="control-panel-inspection-start",
            group="inspection-start",
            exclusive=True,
            exit_on_error=False,
        )

    def _current_inspection_server_status(self) -> InspectionStatus | None:
        """Return viewer status when the application exposes the newer API."""

        status_getter = getattr(self._APPLICATION, "get_inspection_status", None)
        return status_getter() if callable(status_getter) else None

    async def on_unmount(self) -> None:
        """Release GUI-owned observers without cancelling runtime cleanup."""

        self._unmounting = True
        self._configuration_effects.close()
        self._inspection_effects.close()
        self._progress_coordinator.close()
        if self._elapsed_refresh_timer is not None:
            self._elapsed_refresh_timer.pause()
        await self._execution_effects.close()

    def complete_startup_bootstrap(self, force_open: bool = True) -> None:
        """Called by HomeScreen when initial model check/autoload is finished."""
        self._startup_bootstrap_ready = True
        if force_open:
            self.open_startup_configuration_dialog()
        else:
            self._auto_open_startup_dialog_if_needed()

    def prepare_gui_evaluation_continuation(
        self,
        continuation: GuiEvaluationRestartContinuation,
    ) -> None:
        """Restore the transport-neutral fixed evaluation selection."""

        self._evaluation_restart_continuation = continuation
        self._selected_framework_uuid = continuation.framework_uuid
        self._selected_session = continuation.session_name
        self._selected_checkpoint = continuation.checkpoint_name
        self._evaluation_visualization = "gui"
        self._evaluation_pacing = continuation.pacing
        self._evaluation_scenario_index = continuation.scenario_index
        self._evaluation_library_scope = None
        self._evaluation_target = "single"
        self._evaluation_model_uuid = continuation.model_uuid
        self._set_configuration_mode(RLModelRunningModes.EVALUATION)
        self._publish_runtime_context()
        self._update_buttons()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Selectors are managed from configuration dialog."""
        _ = event

    def _update_current_model(self) -> None:
        """Update the current model owned by this control panel."""
        runtime = self._services.runtime.access.get_active_runtime()
        runtime_changed = runtime is not self._current_runtime()
        if runtime_changed:
            expected_model_uuid = self._evaluation_model_uuid
            if not expected_model_uuid or (
                runtime is not None
                and str(
                    getattr(runtime.get_manifest().get_configuration(), "model_uuid", "") or ""
                ) != expected_model_uuid
            ):
                self._evaluation_scenario_index = None
                self._evaluation_library_scope = None
                self._evaluation_target = "single"
                self._evaluation_model_uuid = None
            self._configuration_effects.invalidate()
            self._progress_coordinator.detach()
        snapshot = (
            runtime.get_execution_snapshot()
            if runtime is not None
            else ExecutionSnapshot(None, RLModelState.IDLE, False)
        )
        if runtime_changed or runtime is None:
            self._execution_effects.attach_runtime(runtime, snapshot)
            return
        self._execution_effects.apply_snapshot(runtime, snapshot)

    def _synchronize_execution_snapshot(self) -> ExecutionSnapshot:
        """Return the synchronize execution snapshot value owned by this control panel."""
        return self._execution_effects.synchronize()

    @staticmethod
    def _apply_button_presentation(
        button: Button,
        presentation: ControlPanelButtonPresentation,
    ) -> None:
        """Apply the button presentation transition owned by this control panel."""
        button.display = presentation.display
        button.disabled = presentation.disabled
        if presentation.label is not None:
            button.label = presentation.label

    def _notify_operation(self, notification: UiNotification) -> None:
        """Notify the caller about the operation outcome."""
        self.app.notify(
            notification.message,
            title=notification.title,
            severity=notification.severity,
            timeout=self._NOTIFY_TIMEOUT_SECONDS,
        )

    def set_actions_locked(self, locked: bool) -> None:
        """Set the actions locked owned by this control panel."""
        self._actions_locked = locked
        refresh_buttons = getattr(self, "_update_buttons", None)
        if callable(refresh_buttons):
            refresh_buttons()
            return
        for button_id in (
            "#start",
            "#resume",
            "#pause",
            "#stop",
            "#finish_inspection",
            "#open_native_inspection",
            "#open_analysis_inspection",
        ):
            button = self.query_one(button_id, Button)
            if not hasattr(button, "display"):
                continue
            if button.display:
                button.disabled = locked
            else:
                button.disabled = False

    def _has_valid_runtime_for_continue_training(self) -> bool:
        """Return whether valid runtime for continue training applies to this control panel."""
        return self._services.runtime.status.current_model_has_valid_runtime()

    def _auto_open_startup_dialog_if_needed(self) -> None:
        """Handle the auto open startup dialog if needed operation owned by this control
        panel."""
        if not self._startup_bootstrap_ready:
            return
        if self._startup_dialog_shown_once:
            return
        if self._has_valid_runtime_for_continue_training():
            return
        self._startup_dialog_shown_once = True
        self._configuration_effects.open()

    def open_startup_configuration_dialog(self) -> None:
        """Open the startup configuration dialog workflow owned by this control panel."""
        self._startup_dialog_shown_once = True
        self._configuration_effects.open()

    def open_inspect_configuration_dialog(self) -> None:
        """Open the inspect configuration dialog workflow owned by this control panel."""
        self._startup_dialog_shown_once = True
        self._configuration_effects.open(preferred_mode=RLModelRunningModes.INFERENCE)

    def configuration_effects_available(self) -> bool:
        """Return whether configuration effects available applies to this control panel."""
        return not self._unmounting

    def refresh_configuration_runtime(self) -> None:
        """Refresh the configuration runtime owned by this control panel."""
        self._update_current_model()

    def current_configuration_runtime(self) -> BaseModelRuntime | None:
        """Return the current configuration runtime value owned by this control panel."""
        return self._current_runtime()

    def current_configuration_selection(
        self,
        preferred_mode: RLModelRunningModes | None,
    ) -> ConfigurationDialogSelection:
        """Return the current configuration selection value owned by this control panel."""
        return ConfigurationDialogSelection(
            mode=preferred_mode or self._current_mode,
            framework_uuid=self._resolved_selected_framework_uuid(),
            session_name=self._selected_session,
            checkpoint_name=self._selected_checkpoint,
            expected_model_uuid=None,
            evaluation_visualization=getattr(self, "_evaluation_visualization", "gui"),
            evaluation_pacing=getattr(
                self,
                "_evaluation_pacing",
                "maximum-throughput",
            ),
            evaluation_scenario_index=getattr(
                self,
                "_evaluation_scenario_index",
                None,
            ),
            evaluation_library_scope=getattr(self, "_evaluation_library_scope", None),
            evaluation_target=getattr(self, "_evaluation_target", "single"),
        )

    def apply_configuration_selection(
        self,
        selection: ConfigurationDialogSelection,
    ) -> None:
        """Apply the configuration selection transition owned by this control panel."""
        self._selected_framework_uuid = selection.framework_uuid
        self._selected_session = selection.session_name
        self._selected_checkpoint = selection.checkpoint_name
        self._evaluation_visualization = selection.evaluation_visualization
        self._evaluation_pacing = selection.evaluation_pacing
        self._evaluation_scenario_index = selection.evaluation_scenario_index
        self._evaluation_library_scope = selection.evaluation_library_scope
        self._evaluation_target: EvaluationTarget = (
            "library"
            if selection.evaluation_library_scope is not None
            else selection.evaluation_target
        )
        self._evaluation_model_uuid = selection.expected_model_uuid
        self._set_configuration_mode(selection.mode)

    def _set_configuration_mode(self, mode: RLModelRunningModes) -> None:
        """Set the configuration mode owned by this control panel."""
        if self._current_mode == mode:
            return
        self._current_mode = mode
        self.post_message(self.RunningModeChanged(mode))

    def prepare_new_session_selection(
        self,
        selection: ConfigurationDialogSelection,
    ) -> None:
        """Prepare the new session selection required by this control panel."""
        self._selected_framework_uuid = selection.framework_uuid
        self._set_configuration_mode(selection.mode)

    def configuration_screen_actions(self) -> ConfigurationScreenActionsPort:
        """Return the configuration screen actions value owned by this control panel."""
        return cast(ConfigurationScreenActionsPort, self.screen)

    def push_configuration_dialog(
        self,
        dialog: RunConfigurationDialog,
        callback: Callable[[ConfigurationDialogResult | None], None],
    ) -> None:
        """Push the configuration dialog owned by this control panel."""
        self.app.push_screen(dialog, callback)

    def notify_configuration(self, notification: UiNotification) -> None:
        """Notify the caller about the configuration outcome."""
        self._notify_operation(notification)

    def publish_configuration_context(self) -> None:
        """Publish the configuration context owned by this control panel."""
        self._publish_runtime_context()

    def refresh_configuration_buttons(self) -> None:
        """Refresh the configuration buttons owned by this control panel."""
        self._update_buttons()

    def tensorboard_inspection_is_running(self) -> bool:
        """Return whether tensorboard inspection is running applies to this control panel."""
        return self._inspection_effects.external_running()

    def inspection_effects_available(self) -> bool:
        """Return whether inspection effects available applies to this control panel."""
        return self.is_mounted and not self._unmounting

    def selected_inspection_mode(self) -> RLModelRunningModes:
        """Return the selected inspection mode value owned by this control panel."""
        return self._current_mode

    def selected_inspection_session(self) -> str | None:
        """Return the selected inspection session value owned by this control panel."""
        return self._selected_session

    def selected_inspection_checkpoint(self) -> str | None:
        """Return the selected checkpoint used as initial browser focus."""

        return self._selected_checkpoint

    @staticmethod
    def _viewer_status_text(label: str, viewer: object) -> str:
        """Build one compact independent server status line."""

        state = getattr(getattr(viewer, "state", None), "value", "stopped")
        url = getattr(viewer, "url", None)
        error = getattr(viewer, "error", None)
        detail = url or error or ""
        return f"{label}: {state}" + (f" — {detail}" if detail else "")

    def set_inspection_server_status(self, status: InspectionStatus | None) -> None:
        """Render native and Analysis Dashboard lifecycle state independently."""

        status_container = self.query_one("#inspection_server_status", Vertical)
        actions = self.query_one("#inspection_viewer_container", Horizontal)
        if status is None:
            status_container.display = False
            actions.display = False
            return
        status_container.display = status.target is not None
        actions.display = status.running
        self.query_one("#inspection_native_status", Static).update(
            self._viewer_status_text("Native TensorBoard", status.native)
        )
        self.query_one("#inspection_analysis_status", Static).update(
            self._viewer_status_text("Analysis Dashboard", status.analysis)
        )
        self.query_one("#open_native_inspection", Button).disabled = not status.native.ready
        self.query_one("#open_analysis_inspection", Button).disabled = not status.analysis.ready

    def set_inspection_status(
        self,
        activity: str,
        step: str,
        progress: float | None,
        total: float | None,
    ) -> None:
        """Set the inspection status owned by this control panel."""
        self.set_main_status(activity, step, progress, total)

    def refresh_inspection_runtime(self) -> None:
        """Refresh the inspection runtime owned by this control panel."""
        self.refresh_runtime_context_from_model()

    def refresh_inspection_buttons(self) -> None:
        """Refresh the inspection buttons owned by this control panel."""
        self._update_buttons()

    def reopen_inspection_configuration(self) -> None:
        """Reopen the inspection configuration owned by this control panel."""
        self._configuration_effects.open()

    def _build_tensorboard_status(self) -> str:
        """Build the tensorboard status required by this control panel."""
        return self._APPLICATION.get_tensorboard_status_label(self._selected_session)

    def _publish_runtime_context(self) -> None:
        """Publish the runtime context owned by this control panel."""
        runtime_context = self._services.runtime.context.build_runtime_context_payload(
            selected_session=self._selected_session,
            selected_checkpoint=self._selected_checkpoint,
        )
        self.post_message(
            self.RuntimeContextChanged(
                self._current_mode,
                runtime_context.session_name,
                runtime_context.parent_session_name,
                runtime_context.checkpoint_name,
                self._build_tensorboard_status(),
                runtime_context.session_created_at_raw,
            )
        )

    def set_running_mode(self, mode: RLModelRunningModes) -> None:
        """Explicitly set the running mode from HomeScreen or Select change."""
        self._current_mode = mode
        self._publish_runtime_context()
        self._update_buttons()
        self.post_message(self.RunningModeChanged(self._current_mode))

    def _apply_local_runtime_context(
        self,
        mode: RLModelRunningModes,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> None:
        """Project a runtime selection into the widget without a domain write."""
        if (session_name, checkpoint_name) != (
            self._selected_session,
            self._selected_checkpoint,
        ):
            self._evaluation_scenario_index = None
        self._current_mode = mode
        self._selected_session = session_name
        self._selected_checkpoint = checkpoint_name
        self._publish_runtime_context()
        self._update_buttons()
        self.post_message(self.RunningModeChanged(self._current_mode))

    def refresh_runtime_context_from_model(self) -> None:
        """Refresh the runtime context from model owned by this control
        panel."""
        self._update_current_model()
        if (
            self._current_runtime() is None
            and self._services.runtime.staging.get_selected_manifest() is None
        ):
            self._apply_local_runtime_context(
                RLModelRunningModes.TRAINING,
                None,
                None,
            )
            return

        runtime_selection = self._services.runtime.context.get_current_runtime_selection_state()
        self.set_selected_framework_uuid(self._current_framework_uuid())

        self._apply_local_runtime_context(
            self._current_mode,
            runtime_selection.session_name,
            runtime_selection.checkpoint_name,
        )

    def _update_buttons(self) -> None:
        """Update the buttons owned by this control panel."""
        execution_snapshot = self._synchronize_execution_snapshot()
        runtime_status = self._services.runtime.status
        session_catalog = self._services.catalog.session_checkpoints
        lifecycle_controls_active = bool(
            execution_snapshot.state in ACTIVE_EXECUTION_STATES
            or execution_snapshot.worker_alive
            or self._actions_locked
            or self._inspection_effects.running
        )
        has_valid_model = False
        can_evaluate = False
        can_infer = False
        has_training_session = False
        selected_session_has_checkpoints = False
        selected_session_closed = False
        if not lifecycle_controls_active:
            has_valid_model = runtime_status.current_model_has_valid_structure()
            capability_builder = getattr(
                runtime_status,
                "get_current_model_runtime_capabilities",
                None,
            )
            if callable(capability_builder):
                capabilities = capability_builder()
                can_evaluate = capabilities.has_evaluation_runtime
                can_infer = capabilities.has_inference_runtime
                has_training_session = capabilities.has_training_session
            else:
                can_evaluate = runtime_status.current_model_can_evaluate()
                can_infer = runtime_status.current_model_can_infer()
                has_training_session = runtime_status.current_model_has_training_session()
            selected_session_has_checkpoints = bool(
                self._selected_session
            ) and session_catalog.session_has_checkpoints(self._selected_session)
            selected_session_closed = bool(
                self._selected_session
            ) and session_catalog.session_is_closed_for_training(self._selected_session)
        quick_start = self._execution_effects.build_quick_start_presentation(
            mode=self._current_mode,
            panel_state=execution_snapshot.state,
            has_current_model=(self._current_runtime() is not None or has_valid_model),
            can_evaluate=can_evaluate,
            can_infer=can_infer,
            has_training_session=has_training_session,
            inspect_is_running=self._APPLICATION.get_active_tensorboard_logdir() is not None,
        )
        presentation = self._operation_presenter.build_buttons(
            has_valid_model=has_valid_model,
            actions_locked=self._actions_locked,
            inspection_running=self._inspection_effects.running,
            current_mode=self._current_mode,
            execution_snapshot=execution_snapshot,
            selected_session=self._selected_session,
            selected_session_has_checkpoints=selected_session_has_checkpoints,
            selected_session_closed_for_training=selected_session_closed,
            quick_start_display=quick_start.display,
            quick_start_label=quick_start.label,
            native_gui_restart_required=self.native_gui_restart_required(),
        )

        self._apply_operation_buttons(presentation)
        self.set_inspection_server_status(self._current_inspection_server_status())

    def _apply_operation_buttons(
        self,
        presentation: ControlPanelButtonsPresentation,
    ) -> None:
        """Apply one complete operation-control presentation."""

        self.query_one("#button_container", Horizontal).display = (
            presentation.container.display
        )
        self._apply_button_presentation(
            self.query_one("#start", Button), presentation.start
        )
        self._apply_button_presentation(
            self.query_one("#resume", Button), presentation.resume
        )
        self._apply_button_presentation(
            self.query_one("#pause", Button), presentation.pause
        )
        self._apply_button_presentation(
            self.query_one("#stop", Button), presentation.stop
        )
        self._apply_button_presentation(
            self.query_one("#finish_inspection", Button),
            presentation.finish_inspection,
        )
        self._apply_button_presentation(
            self.query_one("#run_again", Button), presentation.run_again
        )
        self._apply_button_presentation(
            self.query_one("#change_configuration", Button),
            presentation.change_configuration,
        )

    def set_main_status(
        self,
        activity: str,
        step: str = "-",
        progress: float | None = None,
        total: float | None = None,
    ) -> None:
        """Set the main status owned by this control panel."""
        self.query_one("#rl_model_status", Static).update(f"Activity: {activity}")
        self.query_one("#rl_model_substatus", Static).update(f"Step: {step}")
        bar = self.query_one(ProgressBar)
        if total is not None:
            bar.update(total=total)
        if progress is not None:
            bar.update(progress=progress)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this control panel."""
        if self._actions_locked:
            return
        button_id = event.button.id
        if button_id in {"start", "resume"}:
            self._execution_effects.start()
        elif button_id == "pause":
            self._execution_effects.pause()
        elif button_id == "stop":
            self._execution_effects.request_stop()
        elif button_id == "finish_inspection":
            self.finish_tensorboard_inspection()
        elif button_id == "run_again":
            self._request_gui_evaluation_restart("run-again")
        elif button_id == "change_configuration":
            self._request_gui_evaluation_restart("configuration")
        elif button_id == "open_native_inspection":
            self._inspection_effects.open_native()
        elif button_id == "open_analysis_inspection":
            self._inspection_effects.open_analysis()

    def _request_gui_evaluation_restart(self, destination: str) -> None:
        """Delegate one post-run action to the Home-screen effect owner."""

        request = getattr(self.screen, "request_gui_evaluation_restart", None)
        if not callable(request):
            raise RuntimeError("The Home screen cannot restart a GUI evaluation.")
        request(destination)

    def begin_tensorboard_inspection(self, mode: RLModelRunningModes, framework_uuid: str) -> None:
        """Begin the tensorboard inspection owned by this control panel."""
        self._inspection_effects.begin(mode=mode, framework_uuid=framework_uuid)

    def finish_tensorboard_inspection(self, reopen_configuration_dialog: bool = True) -> None:
        """Finish the tensorboard inspection owned by this control panel."""
        self._inspection_effects.finish(
            reopen_configuration_dialog=reopen_configuration_dialog,
        )

    def start_progress_after_configuration_apply(self) -> None:
        """Start the progress after configuration apply workflow owned by this control panel."""
        self._execution_effects.start()

    def dispatch_progress_to_ui(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Dispatch the progress to ui owned by this control panel."""
        if getattr(self, "_unmounting", False):
            return
        try:
            app = self.app
        except NoActiveAppError:
            return
        if getattr(app, "_thread_id", None) == get_ident():
            callback()
            return
        try:
            app.call_from_thread(callback)
        except NoActiveAppError:
            return
        except RuntimeError as exc:
            if "must run in a different thread" in str(exc):
                callback()
                return
            raise

    def accept_progress_event(
        self,
        event: ControlPanelProgressEvent,
    ) -> None:
        """Accept the progress event owned by this control panel."""
        self._execution_effects.accept_progress_event(event)

    def execution_effects_available(self) -> bool:
        """Return whether execution effects available applies to this control panel."""
        return not self._unmounting

    def refresh_execution_runtime(self) -> None:
        """Refresh the execution runtime owned by this control panel."""
        self._update_current_model()

    def selected_execution_mode(self) -> RLModelRunningModes:
        """Return the selected execution mode value owned by this control panel."""
        return self._current_mode

    def selected_execution_session(self) -> str | None:
        """Return the selected execution session value owned by this control panel."""
        return self._selected_session

    def selected_execution_checkpoint(self) -> str | None:
        """Return the selected execution checkpoint value owned by this control panel."""
        return self._selected_checkpoint

    def selected_evaluation_visualization(self) -> EvaluationVisualization:
        """Return the selected evaluation visualization value owned by this control panel."""
        return getattr(self, "_evaluation_visualization", "gui")

    def selected_evaluation_pacing(self) -> EvaluationPacing:
        """Return the selected evaluation pacing value owned by this control panel."""
        return getattr(self, "_evaluation_pacing", "maximum-throughput")

    def selected_evaluation_scenario_index(self) -> int | None:
        """Return the scenario index selected for Evaluation."""

        return getattr(self, "_evaluation_scenario_index", None)

    def selected_evaluation_library_scope(self) -> LibraryEvaluationScope | None:
        """Return a pending library only when an Evaluation run may consume it."""
        if self._current_mode is not RLModelRunningModes.EVALUATION:
            return None
        return getattr(self, "_evaluation_library_scope", None)

    def evaluation_restart_continuation(
        self,
    ) -> GuiEvaluationRestartContinuation | None:
        """Return the evaluation restart continuation value owned by this control panel."""
        return self._evaluation_restart_continuation

    def native_gui_restart_required(self) -> bool:
        """Return whether this process must be replaced before another run."""

        required = getattr(self._APPLICATION, "native_gui_restart_required", None)
        return bool(callable(required) and required())

    def set_post_run_continuation(
        self,
        continuation: GuiEvaluationRestartContinuation | None,
    ) -> None:
        """Store a completed GUI evaluation until the user chooses an action."""

        setter = getattr(
            self._APPLICATION,
            "set_gui_evaluation_post_run_continuation",
            None,
        )
        if callable(setter):
            setter(continuation)

    def set_execution_selection(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> None:
        """Set the execution selection owned by this control panel."""
        if (session_name, checkpoint_name) != (self._selected_session, self._selected_checkpoint):
            self._evaluation_scenario_index = None
        self._selected_session = session_name
        self._selected_checkpoint = checkpoint_name

    def apply_execution_projection(
        self,
        snapshot: ExecutionSnapshot,
        commands_locked: bool,
    ) -> None:
        """Apply the execution projection transition owned by this control panel."""
        self.panel_state = snapshot.state
        self._execution_elapsed.observe(snapshot, now=monotonic())
        self._refresh_execution_elapsed()
        self.post_message(self.ExecutionSnapshotChanged(snapshot, commands_locked))

    def _refresh_execution_elapsed(self) -> None:
        """Render the active run duration beside the progress bar."""

        try:
            elapsed = self.query_one("#progress_elapsed", Static)
        except NoMatches:
            return
        elapsed.update(self._execution_elapsed.display(now=monotonic()))

    def set_execution_status(
        self,
        activity: str,
        step: str,
        progress: float | None = None,
        total: float | None = None,
    ) -> None:
        """Set the execution status owned by this control panel."""
        self.set_main_status(activity, step, progress, total)

    def set_execution_progress(
        self,
        *,
        total: float | None,
        progress: float | None,
        reset_total: bool,
    ) -> None:
        """Set the execution progress owned by this control panel."""
        bar = self.query_one(ProgressBar)
        if reset_total:
            bar.update(total=None, progress=0)
        elif total is not None:
            bar.update(total=total)
        if progress is not None:
            bar.update(progress=progress)

    def render_execution_terminal(self, activity: str, step: str) -> None:
        """Handle the render execution terminal operation owned by this control panel."""
        self.query_one("#rl_model_status", Static).update(f"Activity: {activity}")
        self.query_one("#rl_model_substatus", Static).update(f"Step: {step}")

    def refresh_execution_buttons(self) -> None:
        """Refresh the execution buttons owned by this control panel."""
        self._update_buttons()

    def publish_execution_context(self) -> None:
        """Publish the execution context owned by this control panel."""
        self._publish_runtime_context()

    def notify_execution(
        self,
        message: str,
        *,
        title: str,
        severity: UiSeverity,
    ) -> None:
        """Notify the caller about the execution outcome."""
        self.app.notify(
            message,
            title=title,
            severity=severity,
            timeout=self._NOTIFY_TIMEOUT_SECONDS,
        )

    def request_stop_confirmation(
        self,
        callback: Callable[[bool | None], None],
    ) -> None:
        """Request the stop confirmation owned by this control panel."""
        dialog = secure_stop_training_dialog()  # type: ignore[no-untyped-call]
        self.app.push_screen(dialog, callback)

    def request_model_load(self) -> None:
        """Request the model load owned by this control panel."""
        self.configuration_screen_actions().action_load_model()

    def request_configuration_apply(self) -> None:
        """Request the configuration apply owned by this control panel."""
        self.configuration_screen_actions().action_apply_configuration(start_after_apply=True)

    def request_configuration_dialog(self) -> None:
        """Request the configuration dialog owned by this control panel."""
        self._configuration_effects.open()

    def start_tensorboard_for_execution(self) -> None:
        """Start the tensorboard for execution workflow owned by this control panel."""
        self._inspection_effects.start_for_execution()

    def dispatch_execution_ui(self, callback: Callable[[], None]) -> None:
        """Dispatch the execution ui owned by this control panel."""
        self.dispatch_progress_to_ui(callback)
