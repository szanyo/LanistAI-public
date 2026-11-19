# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own ControlPanel TensorBoard inspection lifecycle effects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from lanista.inspection_lifecycle import InspectionStatus
from lanista.reinforcengine.application_errors import ApplicationError
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.inspection_target_service import InspectionTarget
from lanista.reinforcengine.rl_model.model import RLModelRunningModes


class ControlPanelInspectionViewPort(Protocol):
    """Define the typed boundary for control panel inspection view."""

    def inspection_effects_available(self) -> bool:
        """Return whether inspection effects available applies to this control panel inspection
        view port.
        """
        ...

    def selected_inspection_mode(self) -> RLModelRunningModes:
        """Return the selected inspection mode value provided by this control panel inspection view
        port.
        """
        ...

    def selected_inspection_session(self) -> str | None:
        """Return the selected inspection session value provided by this control panel inspection
        view port.
        """
        ...

    def selected_inspection_checkpoint(self) -> str | None:
        """Return the checkpoint used as the initial inspection focus."""

        ...

    def set_inspection_server_status(self, status: InspectionStatus | None) -> None:
        """Render the two viewer lifecycles independently."""

        ...

    def set_inspection_status(
        self,
        activity: str,
        step: str,
        progress: float | None,
        total: float | None,
    ) -> None:
        """Set the inspection status owned by this control panel inspection view port."""
        ...

    def refresh_inspection_runtime(self) -> None:
        """Refresh the inspection runtime owned by this control panel
        inspection view port."""
        ...

    def refresh_inspection_buttons(self) -> None:
        """Refresh the inspection buttons owned by this control panel
        inspection view port."""
        ...

    def reopen_inspection_configuration(self) -> None:
        """Reopen the inspection configuration owned by this control panel inspection view port."""
        ...


class TensorboardModeRestarter(Protocol):
    """Typed callable boundary for the application TensorBoard lifecycle."""

    def __call__(
        self,
        mode: RLModelRunningModes,
        framework_uuid: str | None = None,
        selected_session: str | None = None,
    ) -> None:
        """Invoke this tensorboard mode restarter as a callable."""
        ...


class ControlPanelInspectionEffectAdapter:
    """Coordinate inspection UI effects without owning execution state."""

    def __init__(
        self,
        *,
        view: ControlPanelInspectionViewPort,
        services: ApplicationServices,
        stop_tensorboard: Callable[[], None],
        restart_tensorboard_for_mode: TensorboardModeRestarter,
        active_logdir: Callable[[], str | None],
        start_inspection: Callable[[InspectionTarget], None] | None = None,
        inspection_status: Callable[[], InspectionStatus | None] | None = None,
        open_native: Callable[[], bool] | None = None,
        open_analysis: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize the control panel inspection effect adapter and its owned state."""
        self._view = view
        self._services = services
        self._stop_tensorboard = stop_tensorboard
        self._restart_tensorboard_for_mode = restart_tensorboard_for_mode
        self._active_logdir = active_logdir
        self._start_inspection = start_inspection
        self._inspection_status = inspection_status
        self._open_native = open_native
        self._open_analysis = open_analysis
        self._active = True
        self._running = False

    @property
    def running(self) -> bool:
        """Return the running exposed by this control panel inspection effect adapter."""
        return self._running

    def activate(self) -> None:
        """Activate the operation owned by this control panel inspection effect adapter."""
        self._active = True

    def close(self) -> None:
        """Close resources owned by this control panel inspection effect adapter."""
        self._active = False

    def external_running(self) -> bool:
        """Return whether external running applies to this control panel inspection effect adapter."""
        status = self._inspection_status() if self._inspection_status is not None else None
        return status.running if status is not None else self._active_logdir() is not None

    def _effects_available(self) -> bool:
        """Return whether effects available applies to this control panel inspection effect
        adapter.
        """
        return self._active and self._view.inspection_effects_available()

    def begin(self, *, mode: RLModelRunningModes, framework_uuid: str) -> None:
        """Begin the operation owned by this control panel inspection effect adapter."""
        if not self._effects_available():
            return
        self._running = True
        self._view.set_inspection_status(
            "Inspection running",
            f"Inspecting {mode.value} / {framework_uuid}",
            100,
            100,
        )
        if self._inspection_status is not None:
            status_setter = getattr(self._view, "set_inspection_server_status", None)
            if callable(status_setter):
                status_setter(self._inspection_status())
        if not self._effects_available():
            return
        self._view.refresh_inspection_buttons()

    def finish(self, *, reopen_configuration_dialog: bool = True) -> None:
        """Finish the operation owned by this control panel inspection effect adapter."""
        if not self._running or not self._effects_available():
            return
        self._running = False
        self._stop_tensorboard()
        if not self._effects_available():
            return
        self._view.set_inspection_status("idle", "-", 0, None)
        status_setter = getattr(self._view, "set_inspection_server_status", None)
        if callable(status_setter):
            status_setter(
                self._inspection_status() if self._inspection_status is not None else None
            )
        if not self._effects_available():
            return
        self._view.refresh_inspection_runtime()
        if not self._effects_available():
            return
        self._view.refresh_inspection_buttons()
        if reopen_configuration_dialog and self._effects_available():
            self._view.reopen_inspection_configuration()

    def start_for_execution(self) -> None:
        """Start the for execution workflow owned by this control panel inspection effect adapter."""
        if not self._effects_available():
            return
        runtime_context = self._services.runtime.context
        context = self._services.runtime.selection.build_tensorboard_restart_context(
            mode=self._view.selected_inspection_mode(),
            selected_session=self._view.selected_inspection_session(),
            configuration=runtime_context.get_active_configuration(),
            current_runtime_framework_uuid=(runtime_context.get_current_framework_uuid()),
        )
        if context is None:
            return
        if self._start_inspection is None:
            self._restart_tensorboard_for_mode(
                mode=context.mode,
                framework_uuid=context.framework_uuid,
                selected_session=context.selected_session,
            )
            if not self._effects_available():
                return
            self._view.refresh_inspection_runtime()
            if not self._effects_available():
                return
            self._view.refresh_inspection_buttons()
        else:
            configuration = runtime_context.get_active_configuration()
            try:
                target = self._services.inspection.targets.ordinary(
                    configuration=configuration,
                    mode=context.mode,
                    framework_uuid=context.framework_uuid,
                    selected_session=context.selected_session,
                    selected_checkpoint=self._view.selected_inspection_checkpoint(),
                )
            except ApplicationError:
                return
            self._start_inspection(target)

    def open_native(self) -> bool:
        """Open native TensorBoard independently of automatic-open policy."""

        return False if self._open_native is None else self._open_native()

    def open_analysis(self) -> bool:
        """Open the Analysis Dashboard independently of automatic-open policy."""

        return False if self._open_analysis is None else self._open_analysis()
