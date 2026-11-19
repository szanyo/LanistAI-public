# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the screen for home."""

import asyncio
import logging
from collections.abc import Callable
from functools import partial

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Footer, Header

from lanista.application import LanistAIApplication
from lanista.foundation.configuration import AppConfiguration
from lanista.gui.adapters.home.gui_evaluation_restart_effect_adapter import (
    GuiEvaluationRestartEffectAdapter,
)
from lanista.gui.adapters.home.home_administration_effect_adapter import (
    HomeAdministrationEffectAdapter,
)
from lanista.gui.adapters.home.home_edit_workflow_adapter import HomeEditWorkflowAdapter
from lanista.gui.adapters.home.home_inspection_effect_adapter import HomeInspectionEffectAdapter
from lanista.gui.adapters.home.home_model_event_adapter import HomeModelEventAdapter
from lanista.gui.adapters.home.home_runtime_effect_adapter import (
    HomeRuntimeControlPanelPort,
    HomeRuntimeEffectAdapter,
)
from lanista.gui.adapters.model_runtime_events import (
    ModelChangedMessage,
    ModelLoadProgressMessage,
    TextualModelRuntimeEventPublisher,
)
from lanista.gui.contracts import UiNotification, UiSeverity
from lanista.gui.predefs.secure_yes_no_dialog_definitions import secure_quit_dialog
from lanista.gui.predefs.yes_no_dialog_definitions import quit_dialog
from lanista.gui.screens.load_model_dialog import LoadModelDialog, LoadModelDialogResult
from lanista.gui.screens.model_creation_screen import ModelCreationResult, ModelCreationScreen
from lanista.gui.screens.settings_screen import SettingsScreen
from lanista.gui.services.async_collection_loader import run_blocking
from lanista.gui.services.home.gui_evaluation_restart_flow_service import (
    GuiEvaluationRestartFlowService,
)
from lanista.gui.services.home.home_compare_flow_service import HomeCompareFlowService
from lanista.gui.services.home.home_configuration_flow_service import (
    ConfigurationApplyPlan,
    HomeConfigurationFlowService,
)
from lanista.gui.services.home.home_edit_flow_service import HomeEditFlowService
from lanista.gui.services.home.home_inspect_flow_service import HomeInspectFlowService
from lanista.gui.services.home.home_model_create_flow_service import HomeModelCreateFlowService
from lanista.gui.services.home.home_model_event_presentation_service import (
    HomeModelEventPresentationService,
)
from lanista.gui.services.home.home_model_load_flow_service import HomeModelLoadFlowService
from lanista.gui.services.home.home_new_session_flow_service import HomeNewSessionFlowService
from lanista.gui.services.home.home_session_detail_flow_service import (
    HomeSessionDetailFlowService,
)
from lanista.gui.services.home.home_startup_selection_flow_service import (
    HomeStartupSelectionFlowService,
)
from lanista.gui.utils.custom_logging import CustomRichLogHandler
from lanista.gui.utils.navigation import route_ids as nav_routes
from lanista.gui.utils.navigation.coordinator import NavigationCoordinator
from lanista.gui.utils.navigation.graph import build_default_navigation_graph
from lanista.gui.widgets.dialogs.about_dialog import AboutDialog
from lanista.gui.widgets.dialogs.checkpoint_configuration_dialog import (
    CheckpointConfigurationDialog,
    CheckpointConfigurationDialogResult,
)
from lanista.gui.widgets.dialogs.comparison_pack_builder_dialog import (
    ComparisonPackBuilderDialog,
    ComparisonPackBuilderDialogResult,
)
from lanista.gui.widgets.dialogs.help_dialog import HelpDialog
from lanista.gui.widgets.dialogs.model_context_dialog import (
    ModelContextDialog,
    ModelContextDialogResult,
)
from lanista.gui.widgets.dialogs.new_session_strategy_dialog import (
    NewSessionStrategyDialog,
    NewSessionStrategyDialogResult,
)
from lanista.gui.widgets.dialogs.secure_yes_no_dialog import SecureYesNoDialog
from lanista.gui.widgets.dialogs.session_configuration_dialog import (
    SessionConfigurationDialog,
    SessionConfigurationDialogResult,
)
from lanista.gui.widgets.dialogs.session_detail_edit_dialog import (
    SessionDetailEditDialog,
    SessionDetailEditDialogResult,
)
from lanista.gui.widgets.dialogs.tensorboard_pack_dialog import (
    TensorboardPackDialog,
    TensorboardPackDialogResult,
)
from lanista.gui.widgets.dialogs.yes_no_dialog import YesNoDialog
from lanista.gui.widgets.fragments.control_panel import ControlPanel
from lanista.gui.widgets.fragments.info_panel import InfoPanel
from lanista.gui.widgets.fragments.logcat_console_panel import LogCatConsolePanel
from lanista.inspection_lifecycle import InspectionStatus
from lanista.locations import get_home_screen_css_path
from lanista.reinforcengine.rl_model.administration_contracts import ModelAdministrationOwner
from lanista.reinforcengine.rl_model.evaluation_restart_contracts import (
    GuiEvaluationRestartContinuation,
    GuiEvaluationRouteRestartContinuation,
)
from lanista.reinforcengine.rl_model.inspection_target_service import InspectionTarget
from lanista.reinforcengine.rl_model.model import RLModelRunningModes

APP_CONFIGURATION = AppConfiguration()

class HomeScreen(Screen[None]):

    """Present home in the Textual interface."""

    CSS_PATH = get_home_screen_css_path()
    _LOGGER_ROOT = logging.getLogger("lanistai")
    _APPLICATION = LanistAIApplication()
    _NOTIFY_TIMEOUT_SECONDS = 30.0

    BINDINGS = [
        Binding(
            key="escape,q,ctrl+c",
            action="request_quit",
            description="Quit",
            key_display="esc",
            tooltip="Quit the application safely, terminating all processes (Esc  | Q  | Ctrl+C)"),
        Binding(
            key="f1,c",
            action="open_configure",
            description="Configure",
            key_display="f1",
            tooltip="Open the run configuration dialog (F1  | C)",
        ),
        Binding(
            key="f2,v",
            action="open_view",
            description="View",
            key_display="f2",
            tooltip="Open model context overview (F2  | V)",
        ),
        Binding(
            key="f3,e",
            action="edit_model_details",
            description="Edit",
            key_display="f3",
            tooltip="Edit model metadata (F3  | E)",
        ),
        Binding(
            key="f4,l",
            action="load_model",
            description="Load",
            key_display="f4",
            tooltip="Open model loader (F4  | L)",
        ),
        Binding(
            key="f8",
            action="open_settings",
            description="Settings",
            key_display="f8",
            tooltip="Open application settings (F8)",
        ),
        Binding(
            key="f9,h,?",
            action="open_help",
            description="Help",
            key_display="f9",
            tooltip="Open help (F9  | H  | ?)",
        ),
        Binding(
            key="f10,a,.",
            action="open_about",
            description="About",
            key_display="f10",
            tooltip="Open about screen (F10  | A  | .)",
        ),
    ]

    def __init__(self) -> None:
        """Initialize the home screen and its owned state."""
        super().__init__()
        self._logger = logging.getLogger(str(__name__).lower())
        self._selected_running_mode: RLModelRunningModes = RLModelRunningModes.TRAINING
        self._model_loading: bool = False
        self._runtime_effect_adapter: HomeRuntimeEffectAdapter | None = None
        self._gui_evaluation_restart_effect_adapter: (
            GuiEvaluationRestartEffectAdapter | None
        ) = None
        self._administration_effect_adapter: HomeAdministrationEffectAdapter | None = None
        self._inspection_effect_adapter: HomeInspectionEffectAdapter | None = None
        self._unmounting: bool = False
        self._compare_flow_service: HomeCompareFlowService | None = None
        self._configuration_flow_service: HomeConfigurationFlowService | None = None
        self._edit_flow_service: HomeEditFlowService | None = None
        self._edit_workflow_adapter: HomeEditWorkflowAdapter | None = None
        self._inspect_flow_service: HomeInspectFlowService | None = None
        self._model_create_flow_service: HomeModelCreateFlowService | None = None
        self._model_load_flow_service: HomeModelLoadFlowService | None = None
        self._model_event_adapter: HomeModelEventAdapter | None = None
        self._model_event_presentation_service: HomeModelEventPresentationService | None = None
        self._new_session_flow_service: HomeNewSessionFlowService | None = None
        self._session_detail_flow_service: HomeSessionDetailFlowService | None = None
        log_widget: LogCatConsolePanel | None = None
        for handler in self._LOGGER_ROOT.handlers:
            if isinstance(handler, CustomRichLogHandler):
                log_widget = handler.log_widget
                break
        self._rich_console = log_widget or LogCatConsolePanel(id="logcat-console")
        self._navigation = NavigationCoordinator(
            screen=self,
            graph=build_default_navigation_graph(),
            opener_map={
                nav_routes.CONFIGURATION_DIALOG: self.action_open_configure,
                nav_routes.LOAD_MODEL_DIALOG: self.action_load_model,
                nav_routes.MODEL_DETAIL_DIALOG: self.action_edit_model_details,
            },
        )

    def _get_compare_flow_service(self) -> HomeCompareFlowService:
        """Return the compare flow service exposed by this home screen."""
        if self._compare_flow_service is None:
            self._compare_flow_service = HomeCompareFlowService(
                self._APPLICATION.get_application_services()
            )
        return self._compare_flow_service

    def _get_configuration_flow_service(self) -> HomeConfigurationFlowService:
        """Return the configuration flow service exposed by this home screen."""
        if self._configuration_flow_service is None:
            self._configuration_flow_service = HomeConfigurationFlowService(
                self._APPLICATION.get_application_services()
            )
        return self._configuration_flow_service

    def _get_administration_effect_adapter(self) -> HomeAdministrationEffectAdapter:
        """Return the administration effect adapter exposed by this home screen."""
        if self._administration_effect_adapter is None:
            self._administration_effect_adapter = HomeAdministrationEffectAdapter(
                view=self,
                services=self._APPLICATION.get_application_services(),
                flow=self._get_configuration_flow_service(),
                is_critical_phase=self._APPLICATION.in_critical_phase,
                selected_mode=lambda: self._selected_running_mode,
            )
        return self._administration_effect_adapter

    def _get_edit_flow_service(self) -> HomeEditFlowService:
        """Return the edit flow service exposed by this home screen."""
        if self._edit_flow_service is None:
            self._edit_flow_service = HomeEditFlowService(
                self._APPLICATION.get_application_services()
            )
        return self._edit_flow_service

    def _get_edit_workflow_adapter(self) -> HomeEditWorkflowAdapter:
        """Return the edit workflow adapter exposed by this home screen."""
        if self._edit_workflow_adapter is None:
            self._edit_workflow_adapter = HomeEditWorkflowAdapter(
                screen=self,
                services=self._APPLICATION.get_application_services(),
                flow_service=self._get_edit_flow_service(),
                navigation=self._navigation,
                is_critical_phase=self._APPLICATION.in_critical_phase,
                stop_tensorboard=self._APPLICATION.stop_tensorboard,
                notify_ui=self._notify_ui,
            )
        return self._edit_workflow_adapter

    def _get_inspect_flow_service(self) -> HomeInspectFlowService:
        """Return the inspect flow service exposed by this home screen."""
        if self._inspect_flow_service is None:
            self._inspect_flow_service = HomeInspectFlowService(
                self._APPLICATION.get_application_services()
            )
        return self._inspect_flow_service

    def _get_inspection_effect_adapter(self) -> HomeInspectionEffectAdapter:
        """Return the inspection effect adapter exposed by this home screen."""
        if self._inspection_effect_adapter is None:
            self._inspection_effect_adapter = HomeInspectionEffectAdapter(
                view=self,
                services=self._APPLICATION.get_application_services(),
                flow=self._get_inspect_flow_service(),
                is_critical_phase=self._APPLICATION.in_critical_phase,
            )
        return self._inspection_effect_adapter

    def _get_model_create_flow_service(self) -> HomeModelCreateFlowService:
        """Return the model create flow service exposed by this home screen."""
        if self._model_create_flow_service is None:
            self._model_create_flow_service = HomeModelCreateFlowService(
                self._APPLICATION.get_application_services()
            )
        return self._model_create_flow_service

    def _get_model_load_flow_service(self) -> HomeModelLoadFlowService:
        """Return the model load flow service exposed by this home screen."""
        if self._model_load_flow_service is None:
            self._model_load_flow_service = HomeModelLoadFlowService(
                self._APPLICATION.get_application_services()
            )
        return self._model_load_flow_service

    def _get_model_event_presentation_service(self) -> HomeModelEventPresentationService:
        """Return the model event presentation service exposed by this home screen."""
        if self._model_event_presentation_service is None:
            self._model_event_presentation_service = HomeModelEventPresentationService(
                self._APPLICATION.get_application_services()
            )
        return self._model_event_presentation_service

    def _get_model_event_adapter(self) -> HomeModelEventAdapter:
        """Return the model event adapter exposed by this home screen."""
        if self._model_event_adapter is None:
            self._model_event_adapter = HomeModelEventAdapter(
                services=self._APPLICATION.get_application_services(),
                presentation=self._get_model_event_presentation_service(),
                control_panel_provider=lambda: self.query_one(
                    "#control_panel",
                    ControlPanel,
                ),
                info_panel_provider=lambda: self.query_one("#info_panel", InfoPanel),
                model_loading_provider=lambda: self._model_loading,
                model_loading_committer=self._set_model_loading,
                startup_autoload_provider=(
                    lambda: self._get_runtime_effect_adapter().startup_in_progress
                ),
                stop_tensorboard=self._APPLICATION.stop_tensorboard,
                notify_ui=self._notify_ui,
            )
        return self._model_event_adapter

    def _set_model_loading(self, loading: bool) -> None:
        """Set the model loading owned by this home screen."""
        self._model_loading = loading

    def _get_runtime_effect_adapter(self) -> HomeRuntimeEffectAdapter:
        """Return the runtime effect adapter exposed by this home screen."""
        if self._runtime_effect_adapter is None:
            services = self._APPLICATION.get_application_services()
            self._runtime_effect_adapter = HomeRuntimeEffectAdapter(
                services=services,
                startup_flow=HomeStartupSelectionFlowService(services),
                control_panel_provider=lambda: self.query_one(
                    "#control_panel",
                    ControlPanel,
                ),
                event_publisher_factory=lambda: TextualModelRuntimeEventPublisher(self),
                model_loading_committer=self._set_model_loading,
                ui_available=lambda: self.is_mounted and not self._unmounting,
                schedule_ui=lambda delay, callback: self.set_timer(delay, callback),
                start_after_activation=self._schedule_start_after_apply,
                activation_failure_notifier=lambda message: self._notify(
                    message,
                    title="Model import failed",
                    severity="error",
                    timeout=self._NOTIFY_TIMEOUT_SECONDS,
                ),
                logger=self._logger,
                restart_continuation_provider=(
                    self._consume_gui_evaluation_restart_continuation
                ),
            )
        return self._runtime_effect_adapter

    def _get_gui_evaluation_restart_effect_adapter(
        self,
    ) -> GuiEvaluationRestartEffectAdapter:
        """Return the Home-owned post-evaluation restart effect."""

        if self._gui_evaluation_restart_effect_adapter is None:
            self._gui_evaluation_restart_effect_adapter = (
                GuiEvaluationRestartEffectAdapter(
                    flow=GuiEvaluationRestartFlowService(),
                    restart_required=self._native_gui_restart_required,
                    replay_provider=(
                        self._APPLICATION.gui_evaluation_post_run_continuation
                    ),
                    owner_identity=(
                        self._APPLICATION.gui_evaluation_post_run_continuation
                    ),
                    push_confirmation=self._push_gui_restart_confirmation,
                    request_restart=self._request_application_restart,
                    notify_failure=lambda message: self._notify(
                        message,
                        title="Restart failed",
                        severity="error",
                    ),
                    logger=self._logger,
                )
            )
        return self._gui_evaluation_restart_effect_adapter

    def _native_gui_restart_required(self) -> bool:
        """Read the optional application capability at the Home boundary."""

        application = getattr(self, "_APPLICATION", None)
        required = getattr(application, "native_gui_restart_required", None)
        return bool(callable(required) and required())

    def _push_gui_restart_confirmation(
        self,
        question: str,
        callback: Callable[[bool | None], None],
    ) -> None:
        """Present the owned Restart/Cancel warning."""

        self.app.push_screen(
            YesNoDialog(
                question=question,
                left_button="Restart",
                left_variant="warning",
                right_button="Cancel",
            ),
            callback,
        )

    def _request_application_restart(self, request: object) -> None:
        """Delegate a validated process replacement to the Textual app."""

        restart = getattr(self.app, "request_application_restart", None)
        if not callable(restart):
            raise RuntimeError("The Textual application cannot restart itself.")
        restart(request)

    def request_gui_evaluation_restart(self, destination: str) -> None:
        """Handle a post-run action selected in the ControlPanel."""

        self._get_gui_evaluation_restart_effect_adapter().request(destination)

    def _consume_gui_evaluation_restart_continuation(
        self,
    ) -> GuiEvaluationRestartContinuation | None:
        """Return the consume gui evaluation restart continuation value owned by this home screen.
        """
        request = self._APPLICATION.consume_startup_restart_request()
        route = GuiEvaluationRouteRestartContinuation.from_restart_request(request)
        if route is not None:
            self._schedule_gui_restart_destination(route.destination)
            return None
        return GuiEvaluationRestartContinuation.from_restart_request(request)

    def _schedule_gui_restart_destination(self, destination: str) -> None:
        """Open a validated route after the fresh Home screen has bootstrapped."""

        actions: dict[str, Callable[[], None]] = {
            "configuration": self.action_open_configure,
            "mode": self.action_open_configure,
            "load-model": self.action_load_model,
            "session": self.action_open_session_selector,
            "checkpoint": self.action_open_checkpoint_selector,
        }
        action = actions.get(destination)
        if action is not None:
            self.set_timer(0.3, action)

    def _get_new_session_flow_service(self) -> HomeNewSessionFlowService:
        """Return the new session flow service exposed by this home screen."""
        if self._new_session_flow_service is None:
            self._new_session_flow_service = HomeNewSessionFlowService(
                self._APPLICATION.get_application_services()
            )
        return self._new_session_flow_service

    def _get_session_detail_flow_service(self) -> HomeSessionDetailFlowService:
        """Return the session detail flow service exposed by this home screen."""
        if self._session_detail_flow_service is None:
            self._session_detail_flow_service = HomeSessionDetailFlowService(
                self._APPLICATION.get_application_services().sessions
            )
        return self._session_detail_flow_service

    def _notify_ui(self, notification: UiNotification) -> None:
        """Notify the caller about the ui outcome."""
        self._notify(
            notification.message,
            title=notification.title,
            severity=notification.severity,
            timeout=self._NOTIFY_TIMEOUT_SECONDS,
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check the action invariant owned by this home screen."""
        _ = parameters
        return not (
            self._APPLICATION.in_critical_phase()
            and action
            in {
                "open_configure",
                "edit_model_details",
                "load_model",
            }
        )

    def _notify(
        self,
        message: str,
        *,
        title: str = "Info",
        severity: UiSeverity = "information",
        timeout: float = _NOTIFY_TIMEOUT_SECONDS,
    ) -> None:
        """Notify the operation owned by this home screen."""
        def _emit() -> None:
            """Publish one record through this home screen."""
            self.notify(message, title=title, severity=severity, timeout=timeout)

        if self.is_mounted:
            self.call_after_refresh(_emit)
        else:
            _emit()

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this home screen."""
        yield Header(show_clock=True)
        with Vertical(id="main_container"):
            yield InfoPanel(id="info_panel")
            yield ControlPanel(id="control_panel")
        yield self._rich_console
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self._unmounting = False
        self._logger.info("HomeScreen mounted.")
        if self._administration_effect_adapter is not None:
            self._administration_effect_adapter.activate()
        if self._inspection_effect_adapter is not None:
            self._inspection_effect_adapter.activate()
        if self._gui_evaluation_restart_effect_adapter is not None:
            self._gui_evaluation_restart_effect_adapter.activate()
        self._get_runtime_effect_adapter().mount()

    async def on_unmount(self) -> None:
        """Cancel and join the screen-owned runtime activation task."""

        self._unmounting = True
        if self._administration_effect_adapter is not None:
            self._administration_effect_adapter.close()
        if self._inspection_effect_adapter is not None:
            self._inspection_effect_adapter.close()
        if self._gui_evaluation_restart_effect_adapter is not None:
            self._gui_evaluation_restart_effect_adapter.close()
        runtime_effect_adapter = self._runtime_effect_adapter
        if runtime_effect_adapter is not None:
            await runtime_effect_adapter.close()
        self._set_model_loading(False)

    def action_request_quit(self) -> None:
        """Action to display the quit dialog."""

        def check_quit(quit: bool | None) -> None:
            """Called when QuitScreen is dismissed."""
            if quit is True:
                self.app.exit()

        self.app.push_screen(secure_quit_dialog() if self._APPLICATION.in_critical_phase() else quit_dialog(), check_quit)

    def action_open_configure(self) -> None:
        """Handle the open configure user action for this home screen."""
        if self._APPLICATION.in_critical_phase():
            self._notify(
                "Configure is disabled while training is running.",
                title="Configure blocked",
                severity="warning",
                timeout=self._NOTIFY_TIMEOUT_SECONDS,
            )
            return
        if HomeScreen._native_gui_restart_required(self):
            HomeScreen.request_gui_evaluation_restart(self, "configuration")
            return
        control_panel = self.query_one("#control_panel", ControlPanel)
        control_panel.open_startup_configuration_dialog()

    def action_open_view(self) -> None:
        """Handle the open view user action for this home screen."""
        application_services = self._APPLICATION.get_application_services()
        if application_services.runtime.staging.get_selected_manifest() is None:
            self._notify(
                "Model context view requires a selected model.",
                title="View blocked",
                severity="warning",
                timeout=self._NOTIFY_TIMEOUT_SECONDS,
            )
            return

        runtime_selection = (
            application_services.runtime.context.get_current_runtime_selection_state()
        )
        current_session_uuid = (
            runtime_selection.session_name
            or application_services.catalog.session_checkpoints.get_latest_session_for_current_model()
        )

        def _handle_context_result(result: ModelContextDialogResult | None) -> None:
            """Handle the context result workflow for this home screen."""
            if result is None:
                return
            if result.action == "edit":
                self.action_edit_model_details()
                return
            if result.action == "sessions":
                self.action_open_session_selector()
                return
            if result.action == "checkpoints":
                self.action_open_checkpoint_selector()

        self.app.push_screen(ModelContextDialog(session_uuid=current_session_uuid), _handle_context_result)

    def action_open_session_selector(self) -> None:
        """Handle the open session selector user action for this home screen."""
        if HomeScreen._native_gui_restart_required(self):
            HomeScreen.request_gui_evaluation_restart(self, "session")
            return
        self._get_administration_effect_adapter().open_sessions()

    def action_open_checkpoint_selector(self) -> None:
        """Handle the open checkpoint selector user action for this home screen."""
        if self._native_gui_restart_required():
            self.request_gui_evaluation_restart("checkpoint")
            return
        self._get_administration_effect_adapter().open_checkpoints()

    def administration_effects_available(self) -> bool:
        """Return whether administration effects available applies to this home screen."""
        return self.is_mounted and not self._unmounting

    def push_session_administration_dialog(
        self,
        dialog: SessionConfigurationDialog,
        callback: Callable[[SessionConfigurationDialogResult | None], None],
    ) -> None:
        """Push the session administration dialog owned by this home screen."""
        self.app.push_screen(dialog, callback)

    def push_checkpoint_administration_dialog(
        self,
        dialog: CheckpointConfigurationDialog,
        callback: Callable[[CheckpointConfigurationDialogResult | None], None],
    ) -> None:
        """Push the checkpoint administration dialog owned by this home screen."""
        self.app.push_screen(dialog, callback)

    def push_administration_confirmation(
        self,
        dialog: SecureYesNoDialog,
        callback: Callable[[bool | None], None],
    ) -> None:
        """Push the administration confirmation owned by this home screen."""
        self.app.push_screen(dialog, callback)

    def refresh_administration_runtime(self) -> None:
        """Refresh the administration runtime owned by this home screen."""
        self.query_one("#control_panel", ControlPanel).refresh_runtime_context_from_model()

    def notify_administration(self, notification: UiNotification) -> None:
        """Notify the caller about the administration outcome."""
        self._notify_ui(notification)

    def open_administration_edit(
        self,
        *,
        session_name: str,
        checkpoint_name: str | None,
        expected_model_uuid: str,
        reopen_opener: Callable[[], None],
    ) -> None:
        """Open the administration edit workflow owned by this home screen."""
        self.action_edit_model_details(
            reopen_opener_on_close=reopen_opener,
            open_details_directly=True,
            selected_session_name=session_name,
            selected_checkpoint_name=checkpoint_name,
            expected_model_uuid=expected_model_uuid,
        )

    def reopen_session_administration(self) -> None:
        """Reopen the session administration owned by this home screen."""
        self.action_open_session_selector()

    def reopen_checkpoint_administration(self) -> None:
        """Reopen the checkpoint administration owned by this home screen."""
        self.action_open_checkpoint_selector()

    def action_open_inspect(self) -> None:
        """Handle the open inspect user action for this home screen."""
        self.action_open_tensorboard_pack_selector()

    def action_open_configuration_inspect(
        self,
        mode: RLModelRunningModes,
        framework_uuid: str | None,
        selected_session: str | None,
        selected_checkpoint: str | None = None,
    ) -> None:
        """Handle the open configuration inspect user action for this home screen."""
        self._get_inspection_effect_adapter().open_configuration(
            mode=mode,
            framework_uuid=framework_uuid,
            selected_session=selected_session,
            selected_checkpoint=selected_checkpoint,
        )

    def action_open_compare_builder(self) -> None:
        """Handle the open compare builder user action for this home screen."""
        compare_flow = self._get_compare_flow_service()
        context = compare_flow.build_builder_context(
            is_critical_phase=self._APPLICATION.in_critical_phase(),
        )
        if context.notification is not None:
            self._notify_ui(context.notification)
            return

        def _on_result(result: ComparisonPackBuilderDialogResult | None) -> None:
            """Handle the result event for this home screen."""
            notification = compare_flow.build_result_notification(result)
            if notification is not None:
                self._notify_ui(notification)

        self.app.push_screen(ComparisonPackBuilderDialog(), _on_result)

    def action_open_help(self) -> None:
        """Handle the open help user action for this home screen."""
        self.app.push_screen(HelpDialog())

    def action_open_settings(self) -> None:
        """Open application-global inspection preferences."""

        self.app.push_screen(SettingsScreen())

    def action_open_about(self) -> None:
        """Handle the open about user action for this home screen."""
        self.app.push_screen(AboutDialog())

    def action_open_tensorboard_pack_selector(self, reopen_startup_dialog: bool = False) -> None:
        """Handle the open tensorboard pack selector user action for this home screen."""
        self._get_inspection_effect_adapter().open_pack(
            reopen_startup_dialog=reopen_startup_dialog,
        )

    def inspection_effects_available(self) -> bool:
        """Return whether inspection effects available applies to this home screen."""
        return self.is_mounted and not self._unmounting

    def push_tensorboard_pack_dialog(
        self,
        dialog: TensorboardPackDialog,
        callback: Callable[[TensorboardPackDialogResult | None], None],
    ) -> None:
        """Push the tensorboard pack dialog owned by this home screen."""
        self.app.push_screen(dialog, callback)

    def start_inspection(
        self,
        target: InspectionTarget,
        callback: Callable[[InspectionStatus], None],
    ) -> None:
        """Start both application-owned viewers outside Textual's event loop."""

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
                status = self._APPLICATION.get_inspection_status()
                if status is None:
                    return
            if self.inspection_effects_available():
                callback(status)

        self.run_worker(
            _start(),
            name="inspection-start",
            group="inspection-start",
            exclusive=True,
            exit_on_error=False,
        )

    def restart_inspection_for_mode(
        self,
        *,
        mode: RLModelRunningModes,
        framework_uuid: str,
        selected_session: str | None,
    ) -> None:
        """Retain the legacy single-view adapter for execution compatibility."""

        self._APPLICATION.restart_tensorboard_for_mode(
            mode=mode,
            framework_uuid=framework_uuid,
            selected_session=selected_session,
        )

    def restart_inspection_for_logdir(
        self,
        logdir: str,
        *,
        online_context: str,
    ) -> None:
        """Retain the guarded legacy logdir adapter for compatibility."""

        self._APPLICATION.restart_tensorboard_for_logdir(
            logdir,
            online_context=online_context,
        )

    def begin_control_panel_inspection(
        self,
        *,
        mode: RLModelRunningModes,
        framework_uuid: str,
    ) -> None:
        """Begin the control panel inspection owned by this home screen."""
        self.query_one("#control_panel", ControlPanel).begin_tensorboard_inspection(
            mode=mode,
            framework_uuid=framework_uuid,
        )

    def refresh_inspection_runtime(self) -> None:
        """Refresh the inspection runtime owned by this home screen."""
        self.query_one("#control_panel", ControlPanel).refresh_runtime_context_from_model()

    def notify_inspection(self, notification: UiNotification) -> None:
        """Notify the caller about the inspection outcome."""
        self._notify_ui(notification)

    def reopen_inspection_configuration(self) -> None:
        """Reopen the inspection configuration owned by this home screen."""
        self._reopen_startup_dialog_later()

    def _reopen_startup_dialog_later(self) -> None:
        """Reopen the startup dialog later owned by this home screen."""
        self._navigation.reopen_route_later(nav_routes.CONFIGURATION_DIALOG)

    def action_create_new_model(self, framework_uuid: str | None = None, reopen_startup_dialog: bool = False) -> None:
        """Action to create a new model."""
        if HomeScreen._native_gui_restart_required(self):
            HomeScreen.request_gui_evaluation_restart(self, "configuration")
            return
        control_panel = self.query_one("#control_panel", ControlPanel)
        model_create_flow = self._get_model_create_flow_service()
        context = model_create_flow.build_dialog_context(
            model_loading=self._model_loading,
            selected_running_mode=self._selected_running_mode,
            requested_framework_uuid=framework_uuid,
            control_panel_framework_uuid=control_panel.get_selected_framework_uuid(),
        )
        if context.blocked_by_loading:
            self._logger.info("Action blocked while model import is in progress: create_new_model")
            return
        if context.blocked_notification is not None:
            self._notify_ui(context.blocked_notification)
            return

        def _handle_created(result: ModelCreationResult | None) -> None:
            """Handle the created workflow for this home screen."""
            if result is None:
                if reopen_startup_dialog:
                    self._navigation.reopen_parent_later(nav_routes.MODEL_CREATION_SCREEN)
                return

            async def _load_created_model() -> None:
                """Load the created model owned by this home screen."""
                creation_result = model_create_flow.handle_creation_result(
                    target_model_folder_path=result.target_model_folder_path,
                    target_model_name=result.target_model_name,
                    creation_strategy=result.creation_strategy,
                    auto_load=result.auto_load,
                    reopen_startup_dialog=reopen_startup_dialog,
                )
                if creation_result.refresh_control_panel:
                    self.query_one("#control_panel", ControlPanel).refresh_runtime_context_from_model()
                if creation_result.notification is not None:
                    self._notify_ui(creation_result.notification)
                if creation_result.reopen_creation_parent:
                    self._navigation.reopen_parent_later(nav_routes.MODEL_CREATION_SCREEN)

            asyncio.create_task(_load_created_model())

        self.app.push_screen(
            ModelCreationScreen(
                framework_uuid=context.selected_framework_uuid,
                application_services=self._APPLICATION.get_application_services(),
            ),
            _handle_created,
        )

    def action_load_model(self, reopen_startup_dialog: bool = False) -> None:
        """Open model load dialog and stage selected model for later confirmation."""
        if self._native_gui_restart_required():
            self.request_gui_evaluation_restart("load-model")
            return
        model_load_flow = self._get_model_load_flow_service()
        context = model_load_flow.build_dialog_context(
            is_critical_phase=self._APPLICATION.in_critical_phase(),
            model_loading=self._model_loading,
        )
        if context.blocked_notification is not None:
            self._notify_ui(context.blocked_notification)
            return

        if context.blocked_by_loading:
            self._logger.info("Action blocked while model import is in progress: load_model")
            return

        def _load_selected(result: LoadModelDialogResult | None) -> None:
            """Load the selected owned by this home screen."""
            load_result = model_load_flow.handle_dialog_result(
                action=getattr(result, "action", None) if result is not None else None,
                model_manifest=getattr(result, "model_manifest", None) if result is not None else None,
                reopen_startup_dialog=reopen_startup_dialog,
            )
            if load_result.canceled:
                if load_result.reopen_load_parent:
                    self._navigation.reopen_parent_later(nav_routes.LOAD_MODEL_DIALOG)
                return

            self.query_one("#control_panel", ControlPanel).refresh_runtime_context_from_model()

            if load_result.open_edit_details:
                self.action_edit_model_details(
                    reopen_route_on_close=nav_routes.LOAD_MODEL_DIALOG,
                    reopen_opener_on_close=lambda: self.action_load_model(
                        reopen_startup_dialog=reopen_startup_dialog,
                    ),
                )
                return

            if load_result.notification is not None:
                self._notify_ui(load_result.notification)
            if load_result.reopen_configuration:
                self._navigation.reopen_route_later(nav_routes.CONFIGURATION_DIALOG)

        self.app.push_screen(LoadModelDialog(current_model_uuid=context.current_model_uuid), _load_selected)

    def action_apply_configuration(
        self,
        start_after_apply: bool = False,
        expected_model_uuid: str | None = None,
    ) -> None:
        """Handle the apply configuration user action for this home screen."""
        normalized_model_uuid = str(expected_model_uuid or "").strip()
        if normalized_model_uuid and not (
            self._APPLICATION.get_application_services()
            .administration.models.is_owner_current(
                ModelAdministrationOwner(normalized_model_uuid)
            )
        ):
            return
        control_panel = self.query_one("#control_panel", ControlPanel)
        apply_plan = self._get_configuration_flow_service().build_apply_plan(
            model_loading=self._model_loading,
            selected_mode=control_panel.get_selected_mode(),
            selected_framework_uuid=control_panel.get_selected_framework_uuid(),
        )
        if apply_plan.blocked_by_loading:
            return
        self._present_configuration_apply_plan(apply_plan, control_panel)
        if apply_plan.import_required:
            self._schedule_runtime_activation(
                apply_plan,
                control_panel,
                start_after_apply=start_after_apply,
            )
            return
        self._continue_after_reused_configuration(
            apply_plan,
            control_panel,
            start_after_apply=start_after_apply,
        )

    def _present_configuration_apply_plan(
        self,
        apply_plan: ConfigurationApplyPlan,
        control_panel: ControlPanel,
    ) -> None:
        """Present the configuration apply plan owned by this home screen."""
        if apply_plan.selection_log_message is not None:
            self._logger.info(apply_plan.selection_log_message)
        if apply_plan.result_log_message is not None:
            self._logger.info(apply_plan.result_log_message)
        if apply_plan.notification is not None:
            self._notify_ui(apply_plan.notification)
        if apply_plan.refresh_control_panel:
            control_panel.refresh_runtime_context_from_model()

    def _continue_after_reused_configuration(
        self,
        apply_plan: ConfigurationApplyPlan,
        control_panel: ControlPanel,
        *,
        start_after_apply: bool,
    ) -> None:
        """Handle the continue after reused configuration operation owned by this home
screen."""
        if not start_after_apply or apply_plan.notification is not None:
            return
        if self._APPLICATION.get_application_services().runtime.status.has_loaded_runtime():
            self._schedule_start_after_apply(control_panel)
            return
        self._logger.error(
            "Configuration apply completed without importing a model, "
            "but no runtime model is loaded. Automatic start canceled."
        )
        self._notify(
            (
                "The model is selected, but its runtime is not loaded. "
                "Automatic start was canceled to prevent a retry loop."
            ),
            title="Model import required",
            severity="error",
            timeout=self._NOTIFY_TIMEOUT_SECONDS,
        )

    def _schedule_runtime_activation(
        self,
        apply_plan: ConfigurationApplyPlan,
        control_panel: ControlPanel,
        *,
        start_after_apply: bool,
    ) -> None:
        """Schedule the runtime activation owned by this home screen."""
        self._get_runtime_effect_adapter().schedule_activation(
            apply_plan,
            control_panel,
            start_after_apply=start_after_apply,
        )

    def _schedule_start_after_apply(
        self,
        control_panel: HomeRuntimeControlPanelPort,
    ) -> None:
        """Schedule the start after apply owned by this home screen."""
        self.set_timer(0.01, control_panel.start_progress_after_configuration_apply)

    def action_create_new_session(
        self,
        reopen_startup_dialog: bool = False,
        selected_running_mode: RLModelRunningModes | None = None,
        selected_framework_uuid: str | None = None,
        selected_session_name: str | None = None,
        expected_model_uuid: str | None = None,
    ) -> None:
        """Action to create a new session for selected model."""
        if HomeScreen._native_gui_restart_required(self):
            HomeScreen.request_gui_evaluation_restart(self, "session")
            return
        control_panel = self.query_one("#control_panel", ControlPanel)
        new_session_flow = self._get_new_session_flow_service()
        context = new_session_flow.build_dialog_context(
            model_loading=self._model_loading,
            selected_running_mode=selected_running_mode or self._selected_running_mode,
            control_panel_session_uuid=selected_session_name,
            control_panel_framework_uuid=selected_framework_uuid or control_panel.get_selected_framework_uuid(),
            expected_model_uuid=expected_model_uuid,
        )
        if context.blocked_by_loading:
            self._logger.info("Action blocked while model import is in progress: create_new_session")
            return
        if context.blocked_notification is not None:
            self._notify_ui(context.blocked_notification)
            return

        def _on_strategy_selected(result: NewSessionStrategyDialogResult | None) -> None:
            """Handle the strategy selected event for this home screen."""
            if result is None:
                cancel_result = new_session_flow.build_cancel_result(
                    reopen_startup_dialog=reopen_startup_dialog,
                )
                if cancel_result.reopen_session_parent:
                    self._navigation.reopen_parent_later(nav_routes.NEW_SESSION_DIALOG)
                return

            create_result = new_session_flow.create_session(
                strategy=result.strategy,
                parent_session_uuid=result.parent_session_uuid,
                parent_checkpoint_name=result.parent_checkpoint_name,
                framework_uuid=context.framework_uuid,
                session_display_name=result.session_name,
                session_description=result.session_description,
                reopen_startup_dialog=reopen_startup_dialog,
                expected_model_uuid=context.expected_model_uuid,
            )

            if create_result.refresh_control_panel:
                control_panel.refresh_runtime_context_from_model()
            if create_result.notification is not None:
                self._notify_ui(create_result.notification)
            if create_result.reopen_session_parent:
                self._navigation.reopen_parent_later(nav_routes.NEW_SESSION_DIALOG)

        self.app.push_screen(
            NewSessionStrategyDialog(
                current_session_uuid=context.current_session_uuid,
                current_session_display_name=context.current_session_display_name,
                framework_uuid=context.framework_uuid,
            ),
            _on_strategy_selected,
        )

    def action_edit_model_details(
        self,
        reopen_route_on_close: str | None = None,
        reopen_opener_on_close: Callable[[], None] | None = None,
        open_details_directly: bool = False,
        selected_session_name: str | None = None,
        selected_checkpoint_name: str | None = None,
        expected_model_uuid: str | None = None,
    ) -> None:
        """Start a guarded metadata-edit workflow for the selected model."""

        if HomeScreen._native_gui_restart_required(self):
            self.request_gui_evaluation_restart("configuration")
            return

        self._get_edit_workflow_adapter().start(
            reopen_route=reopen_route_on_close,
            reopen_opener=reopen_opener_on_close,
            open_details_directly=open_details_directly,
            selected_session_name=selected_session_name,
            selected_checkpoint_name=selected_checkpoint_name,
            expected_model_uuid=expected_model_uuid,
        )

    def action_edit_session_details(
        self,
        session_uuid: str | None,
        reopen_opener_on_close: Callable[[], None] | None = None,
    ) -> None:
        """Handle the edit session details user action for this home screen."""
        if self._native_gui_restart_required():
            self.request_gui_evaluation_restart("session")
            return

        def _reopen_on_close() -> None:
            """Reopen the on close owned by this home screen."""
            if reopen_opener_on_close is not None:
                self.set_timer(0.15, reopen_opener_on_close)

        session_detail_flow = self._get_session_detail_flow_service()
        context = session_detail_flow.build_context(
            is_critical_phase=self._APPLICATION.in_critical_phase(),
            session_uuid=session_uuid,
        )
        if context.blocked_notification is not None:
            self._notify_ui(context.blocked_notification)
            _reopen_on_close()
            return
        if context.session_uuid is None:
            _reopen_on_close()
            return

        def _handle_session_detail_result(result: SessionDetailEditDialogResult | None) -> None:
            """Handle the session detail result workflow for this home screen."""
            if result is None:
                _reopen_on_close()
                return

            if result.action != "save":
                _reopen_on_close()
                return

            save_result = session_detail_flow.save_session_details(
                session_uuid=result.session_uuid,
                session_name=result.session_name,
                session_description=result.session_description,
            )
            if not save_result.success:
                self._notify_ui(save_result.notification)
                _reopen_on_close()
                return

            self.query_one("#control_panel", ControlPanel).refresh_runtime_context_from_model()
            self._notify_ui(save_result.notification)
            _reopen_on_close()

        self.app.push_screen(
            SessionDetailEditDialog(session_uuid=context.session_uuid),
            _handle_session_detail_result,
        )

    @on(ModelChangedMessage)
    def on_model_changed(self, event: ModelChangedMessage) -> None:
        """Project one runtime-slot event through the typed Home adapter."""

        if self._administration_effect_adapter is not None:
            self._administration_effect_adapter.invalidate()
        if self._inspection_effect_adapter is not None:
            self._inspection_effect_adapter.invalidate()
        self._get_model_event_adapter().handle(event)

    @on(ModelLoadProgressMessage)
    def on_model_load_progress(self, event: ModelLoadProgressMessage) -> None:
        """Handle the model load progress event for this home screen."""
        control_panel = self.query_one("#control_panel", ControlPanel)
        control_panel.set_main_status(event.activity, event.step, event.progress, event.total)


    def on_control_panel_execution_snapshot_changed(
        self,
        event: ControlPanel.ExecutionSnapshotChanged,
    ) -> None:
        """Receive the full execution projection, including same-state updates."""

        _ = event

    def on_control_panel_running_mode_changed(self, event: ControlPanel.RunningModeChanged) -> None:
        """Track current phase mode to enforce UI restrictions."""
        previous_mode = self._selected_running_mode
        self._selected_running_mode = event.mode
        if previous_mode != event.mode:
            self._APPLICATION.stop_tensorboard()

    def on_control_panel_runtime_context_changed(self, event: ControlPanel.RuntimeContextChanged) -> None:
        """Handle the control panel runtime context changed event for this home screen."""
        try:
            self._get_model_event_adapter().handle_runtime_context(event)
        except NoMatches:
            self._logger.debug(
                "Ignored runtime context update because the info panel is unavailable."
            )
