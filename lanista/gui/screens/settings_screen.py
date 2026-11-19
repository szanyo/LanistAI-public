# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present application-global inspection settings."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, Switch

from lanista.application import LanistAIApplication
from lanista.application_restart_service import ApplicationRestartRequest
from lanista.gui.widgets.dialogs.button_templates import (
    cancel_button,
    dialog_action_row,
    primary_button,
)
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.dialogs.yes_no_dialog import YesNoDialog
from lanista.gui.widgets.modal_footer import ModalFooter


class SettingsScreen(HelpEnabledModalMixin, ModalScreen[None]):
    """Edit persistent application preferences without model metadata writes."""

    _HELP_TOPIC = "settings"
    _APPLICATION = LanistAIApplication()
    _gpu_change_pending = False

    BINDINGS = [
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "dismiss(None)", "Close", key_display="esc"),
        Binding("f8", "dismiss(None)", "Close settings", key_display="f8"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the standard modal frame and scrollable settings body."""

        preferences = self._APPLICATION.get_inspection_preferences()
        with Vertical(id="settings_dialog_container") as container:
            container.border_title = "Settings"
            with (
                ScrollableContainer(id="settings_dialog_scroll"),
                Vertical(id="settings_dialog_content"),
            ):
                yield Static(
                    "Configure startup, training device, and inspection behavior. "
                    "GPU changes require a confirmed restart.",
                    id="settings_preferences_hint",
                )
                yield Static("Startup", classes="settings-section-title")
                yield self._switch_row(
                    "Show splash screen at startup",
                    self._APPLICATION.get_splash_screen_enabled(),
                    "settings_splash_screen",
                )
                yield self._switch_row(
                    "Maximize terminal at startup (best effort)",
                    self._APPLICATION.get_maximize_terminal_enabled(),
                    "settings_maximize_terminal",
                )
                yield self._switch_row(
                    "Fullscreen terminal at startup (best effort)",
                    self._APPLICATION.get_fullscreen_enabled(),
                    "settings_fullscreen",
                )
                yield Static("Training device", classes="settings-section-title")
                yield Static(
                    "Use GPU for SB3 and Ray training. Changing this setting "
                    "requires an immediate application restart.",
                    id="settings_gpu_hint",
                )
                yield self._switch_row(
                    "Use GPU for training",
                    self._APPLICATION.get_training_gpu_enabled(),
                    "settings_use_gpu",
                )
                yield Static("Inspection", classes="settings-section-title")
                yield Static(
                    "Inspection servers start as needed. These switches control "
                    "automatic browser opening.",
                    id="settings_inspection_hint",
                )
                yield self._switch_row(
                    "Auto-open native TensorBoard",
                    preferences.auto_open_native_tensorboard,
                    "settings_auto_open_native",
                )
                yield self._switch_row(
                    "Auto-open Analysis Dashboard",
                    preferences.auto_open_analysis_dashboard,
                    "settings_auto_open_analysis",
                )
                yield Static(
                    "The splash setting applies after restarting LanistAI. Inspection "
                    "settings apply to the next inspection target.",
                    id="settings_persistence_hint",
                )
                yield Static(
                    "Open a ready inspection page manually. Native TensorBoard becomes "
                    "available after an inspection target is selected.",
                    id="settings_inspection_open_hint",
                )
                yield self._inspection_open_actions()
            yield dialog_action_row(
                cancel_button(
                    "Close",
                    id="settings_dialog_close",
                ),
                id="settings_dialog_actions",
            )
            yield self._modal_footer()

    @staticmethod
    def _switch_row(label: str, value: bool, switch_id: str) -> Horizontal:
        """Build one consistently structured settings switch row."""

        return Horizontal(
            Static(label, classes="settings-switch-label"),
            Switch(value=value, animate=False, id=switch_id),
            classes="settings-switch-row",
        )

    def _modal_footer(self) -> ModalFooter:
        """Build the Settings shortcut footer."""

        return ModalFooter.from_binding_rows(
            self.BINDINGS,
            [["open_help_dialog", "dismiss(None)"]],
            id="settings_modal_footer",
            label_overrides={"dismiss(None)": "Close"},
        )

    def _inspection_open_actions(self) -> Horizontal:
        """Build manual-open actions from one lifecycle status snapshot."""

        status = self._APPLICATION.get_inspection_status()
        native_ready = bool(status and status.native.ready)
        analysis_ready = bool(status and status.analysis.ready)
        return dialog_action_row(
            primary_button(
                "Open native TensorBoard",
                id="settings_open_native_tensorboard",
                disabled=not native_ready,
            ),
            primary_button(
                "Open Analysis Dashboard",
                id="settings_open_analysis_dashboard",
                disabled=not analysis_ready,
            ),
            id="settings_inspection_open_actions",
        )

    @on(Button.Pressed, "#settings_dialog_close")
    def _close_settings(self) -> None:
        """Close Settings from its visible About-style action button."""

        self.dismiss(None)

    @on(Switch.Changed, "#settings_splash_screen")
    def _set_splash_screen(self, event: Switch.Changed) -> None:
        """Persist whether the splash screen appears at next startup."""

        self._APPLICATION.set_splash_screen_enabled(event.value)

    @on(Switch.Changed, "#settings_maximize_terminal")
    def _set_maximize_terminal(self, event: Switch.Changed) -> None:
        """Persist whether terminal maximization is requested at next startup."""

        self._APPLICATION.set_maximize_terminal_enabled(event.value)

    @on(Switch.Changed, "#settings_fullscreen")
    def _set_fullscreen(self, event: Switch.Changed) -> None:
        """Persist whether terminal fullscreen mode is requested at next startup."""

        self._APPLICATION.set_fullscreen_enabled(event.value)

    @on(Switch.Changed, "#settings_use_gpu")
    def _change_training_gpu(self, event: Switch.Changed) -> None:
        """Confirm a process restart before changing the saved GPU choice."""

        previous = self._APPLICATION.get_training_gpu_enabled()
        requested = bool(event.value)
        if requested == previous or self._gpu_change_pending:
            return
        gpu_switch = self.query_one("#settings_use_gpu", Switch)
        if self._APPLICATION.in_critical_phase():
            gpu_switch.value = previous
            self.app.notify(
                "Stop the current run before changing GPU use.",
                severity="warning",
            )
            return

        self._gpu_change_pending = True
        gpu_switch.disabled = True

        def resolve(confirmed: bool | None) -> None:
            """Apply the saved choice only when restart was accepted."""

            self._gpu_change_pending = False
            gpu_switch.disabled = False
            if confirmed is not True:
                gpu_switch.value = previous
                return
            try:
                restart = getattr(self.app, "request_application_restart", None)
                if not callable(restart):
                    raise RuntimeError("The application cannot restart itself.")
                self._APPLICATION.set_training_gpu_enabled(requested)
                restart(
                    ApplicationRestartRequest(
                        reason="training-gpu-change",
                        continuation={"previous_use_gpu": previous},
                    )
                )
            except Exception as error:
                self._APPLICATION.set_training_gpu_enabled(previous)
                gpu_switch.value = previous
                self.app.notify(
                    f"GPU setting was not changed: {error}",
                    severity="error",
                )

        self.app.push_screen(
            YesNoDialog(
                question=(
                    "LanistAI must restart to change GPU use. "
                    "Restart now?"
                ),
                left_button="Restart now",
                left_variant="primary",
                right_button="Keep current setting",
            ),
            resolve,
        )

    @on(Switch.Changed, "#settings_auto_open_native")
    def _set_native_auto_open(self, event: Switch.Changed) -> None:
        """Persist the native TensorBoard browser preference immediately."""

        self._APPLICATION.set_auto_open_native_tensorboard(event.value)

    @on(Switch.Changed, "#settings_auto_open_analysis")
    def _set_analysis_auto_open(self, event: Switch.Changed) -> None:
        """Persist the Analysis Dashboard browser preference immediately."""

        self._APPLICATION.set_auto_open_analysis_dashboard(event.value)

    @on(Button.Pressed, "#settings_open_native_tensorboard")
    def _open_native_tensorboard(self) -> None:
        """Open the current native TensorBoard URL without changing preferences."""

        if not self._APPLICATION.open_native_inspection():
            self.app.notify("Native TensorBoard is not ready.", severity="warning")

    @on(Button.Pressed, "#settings_open_analysis_dashboard")
    def _open_analysis_dashboard(self) -> None:
        """Open the application-owned Analysis URL without changing preferences."""

        if not self._APPLICATION.open_analysis_inspection():
            self.app.notify("Analysis Dashboard is not ready.", severity="warning")


__all__ = ["SettingsScreen"]
