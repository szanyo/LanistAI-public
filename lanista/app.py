# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Start and coordinate the LanistAI application."""

import logging
from collections.abc import Iterable

from textual.app import App, SystemCommand
from textual.binding import Binding
from textual.screen import Screen

from lanista.application import LanistAIApplication
from lanista.application_bootstrap import APP_CONFIGURATION
from lanista.gui.adapters.terminal_window_adapter import (
    request_terminal_fullscreen,
    request_terminal_maximize,
)
from lanista.gui.adapters.textual_capture_stream_adapter import (
    TextualCaptureStreamAdapter,
)
from lanista.gui.predefs.secure_yes_no_dialog_definitions import secure_quit_dialog
from lanista.gui.predefs.yes_no_dialog_definitions import quit_dialog
from lanista.gui.screens.home_screen import HomeScreen
from lanista.gui.screens.settings_screen import SettingsScreen
from lanista.gui.screens.splash_screen import SplashArtScreen
from lanista.gui.widgets.dialogs.about_dialog import AboutDialog
from lanista.gui.widgets.dialogs.developer_tools_dialog import DeveloperToolsDialog
from lanista.gui.widgets.dialogs.help_dialog import HelpDialog
from lanista.locations import get_app_css_path
from lanista.reinforcengine.application_shutdown_service import (
    ApplicationShutdownReport,
    ApplicationShutdownService,
)

LOGGER = logging.getLogger(__name__)


def _stop_optional_ray_runtime() -> None:
    """Stop an initialized Ray process when the dependency is available."""

    try:
        import ray
    except ModuleNotFoundError as error:
        if error.name == "ray":
            LOGGER.debug("Ray is unavailable during runtime shutdown.")
            return
        raise
    if ray.is_initialized():
        ray.shutdown()


class LanistAIApp(App[None]):
    """Coordinate the LanistAI application lifecycle."""

    _APPLICATION = LanistAIApplication()

    TITLE = "LanistAI"
    SUB_TITLE = (
        "A modular framework for reinforcement learning training, "
        "evaluation, and inference"
    )

    CSS_PATH = get_app_css_path()

    BINDINGS = [
        Binding("ctrl+c", "help_quit", show=False, system=True),
        Binding("ctrl+q", "help_quit", show=False),
    ]

    MODES = {
        "splash": SplashArtScreen,
        "home": HomeScreen,
        "help": HelpDialog,
        "about": AboutDialog,
    }

    DEFAULT_MODE = "splash" if APP_CONFIGURATION.gui.splash_screen else "home"

    ENABLE_COMMAND_PALETTE = APP_CONFIGURATION.sudo

    _shutdown_failures: tuple[str, ...] = ()
    _shutdown_report: ApplicationShutdownReport | None = None
    _last_recoverable_ui_failure: str | None = None

    def __init__(self) -> None:
        """Initialize the LanistAI app and its owned state."""
        super().__init__()
        self._capture_stdout = TextualCaptureStreamAdapter(  # type: ignore[assignment]
            self._capture_stdout,
            self._original_stdout,
        )
        self._capture_stderr = TextualCaptureStreamAdapter(  # type: ignore[assignment]
            self._capture_stderr,
            self._original_stderr,
        )

    def action_open_developer_tools(self) -> None:
        """Handle the open developer tools user action for this LanistAI app."""
        self.push_screen(DeveloperToolsDialog())

    def action_open_settings(self) -> None:
        """Open application-global preferences."""

        self.push_screen(SettingsScreen())

    def on_mount(self) -> None:
        """Surface a failed process replacement after the UI remounts."""

        failure = self._APPLICATION.consume_restart_failure()
        if failure is None:
            return
        self.notify(
            f"LanistAI could not restart: {failure}. "
            "The current process remains guarded; choose an action to retry or exit.",
            title="Restart failed",
            severity="error",
            timeout=30.0,
        )

    def _handle_exception(self, error: Exception) -> None:
        """Keep the interactive shell alive after a recoverable UI failure.

        Textual's default boundary always terminates the application. Once the
        HomeScreen is mounted, execution and presentation callbacks must leave
        the LogCat console available so the user can inspect the failure.
        Bootstrap and screen-composition failures still use Textual's fatal
        boundary because no reliable interactive shell exists for them.
        """

        if not self._interactive_error_recovery_available():
            super()._handle_exception(error)
            return
        failure = f"{type(error).__name__}: {error}"
        self._last_recoverable_ui_failure = failure
        LOGGER.error(
            "Unhandled Textual callback failed; keeping the application open.",
            exc_info=(type(error), error, error.__traceback__),
        )
        try:
            self.notify(
                "An application callback failed. The application remains open; "
                "inspect LogCat for the traceback.",
                title="Application error",
                severity="error",
                timeout=30.0,
            )
        except Exception:
            LOGGER.exception(
                "Could not display the recoverable UI failure notification."
            )

    def _interactive_error_recovery_available(self) -> bool:
        """Return whether a mounted HomeScreen can still present diagnostics."""

        try:
            screens = self.screen_stack
        except Exception:
            return False
        return any(
            isinstance(screen, HomeScreen) and screen.is_mounted
            for screen in screens
        )

    def get_system_commands(
        self,
        screen: Screen[object],
    ) -> Iterable[SystemCommand]:
        """Return the system commands exposed by this LanistAI app."""
        yield from super().get_system_commands(screen)
        yield SystemCommand(
            "Developer Tools",
            "Open the stylesheet showcase and design template atlas",
            self.action_open_developer_tools,
        )
        yield SystemCommand(
            "Settings",
            "Configure inspection browser-opening preferences",
            self.action_open_settings,
        )

    async def _shutdown_runtime(self) -> ApplicationShutdownReport:
        """Await runtime quiescence before process-global cleanup."""

        shutdown_service = ApplicationShutdownService(
            tensorboard_stopper=self._APPLICATION.shutdown_inspection,
            active_runtime_provider=lambda: (
                self._APPLICATION.get_application_services().runtime.access.get_active_runtime()
            ),
            ray_stopper=_stop_optional_ray_runtime,
            logger=LOGGER,
        )
        report = await shutdown_service.shutdown()
        self._shutdown_report = report
        self._shutdown_failures = report.failure_steps
        return report

    def action_help_quit(self) -> None:
        """Handle the help quit user action for this LanistAI app."""
        def check_quit(quit: bool | None) -> None:
            """Check the quit invariant owned by this LanistAI app."""
            if quit is True:
                self.exit()

        self.push_screen(
            secure_quit_dialog() if self._APPLICATION.in_critical_phase() else quit_dialog(),
            check_quit,
        )

    async def on_unmount(self) -> None:
        """Handle the unmount event for this LanistAI app."""
        if self._APPLICATION.restart_pending():
            LOGGER.info(
                "Skipping in-process runtime cleanup because a full process-image "
                "restart owns native cleanup."
            )
            return
        await self._shutdown_runtime()

    def request_application_restart(self, request: object) -> None:
        """Register one application-owned restart and leave the Textual loop."""

        from lanista.application_restart_service import ApplicationRestartRequest

        if not isinstance(request, ApplicationRestartRequest):
            raise TypeError("Application restart requires a typed request.")
        self._APPLICATION.request_restart(request)
        self.exit()


def main() -> None:
    """Run the LanistAI Textual application."""

    application = LanistAIApplication()
    LanistAIApp._APPLICATION = application
    if application.get_fullscreen_enabled() is True:
        request_terminal_fullscreen()
    elif application.get_maximize_terminal_enabled() is True:
        request_terminal_maximize()
    with application:
        while True:
            LanistAIApp().run()
            if not application.restart_pending():
                break
            try:
                application.replace_process_image()
            except Exception as error:
                LOGGER.exception(
                    "Application process replacement failed; remounting the UI."
                )
                application.record_restart_failure(
                    f"{type(error).__name__}: {error}"
                )
                application.clear_failed_restart_request()
            else:
                return


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
