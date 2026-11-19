# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own Home TensorBoard inspection callbacks across one screen/model lifetime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from lanista.gui.contracts import UiNotification
from lanista.gui.services.home.home_inspect_flow_service import (
    HomeInspectFlowService,
    TensorboardPackOpenPlan,
    TensorboardPackSelection,
)
from lanista.gui.widgets.dialogs.tensorboard_pack_dialog import (
    TensorboardPackDialog,
    TensorboardPackDialogResult,
)
from lanista.inspection_lifecycle import InspectionStatus
from lanista.reinforcengine.rl_model.administration_contracts import ModelAdministrationOwner
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.inspection_target_service import InspectionTarget
from lanista.reinforcengine.rl_model.model import RLModelRunningModes


class HomeInspectionViewPort(Protocol):
    """Define the typed boundary for home inspection view."""

    def inspection_effects_available(self) -> bool:
        """Return whether inspection effects available applies to this home inspection view port."""
        ...

    def push_tensorboard_pack_dialog(
        self,
        dialog: TensorboardPackDialog,
        callback: Callable[[TensorboardPackDialogResult | None], None],
    ) -> None:
        """Push the tensorboard pack dialog owned by this home inspection view port."""
        ...

    def start_inspection(
        self,
        target: InspectionTarget,
        callback: Callable[[InspectionStatus], None],
    ) -> None:
        """Start both viewers without blocking Textual's event loop."""
        ...

    def begin_control_panel_inspection(
        self,
        *,
        mode: RLModelRunningModes,
        framework_uuid: str,
    ) -> None:
        """Begin the control panel inspection owned by this home inspection view port."""
        ...

    def refresh_inspection_runtime(self) -> None:
        """Refresh the inspection runtime owned by this home inspection view
port."""
        ...

    def notify_inspection(self, notification: UiNotification) -> None:
        """Notify the caller about the inspection outcome."""
        ...

    def reopen_inspection_configuration(self) -> None:
        """Reopen the inspection configuration owned by this home inspection view port."""
        ...


class HomeInspectionEffectAdapter:
    """Guard TensorBoard restarts by mount generation and captured model owner."""

    def __init__(
        self,
        *,
        view: HomeInspectionViewPort,
        services: ApplicationServices,
        flow: HomeInspectFlowService,
        is_critical_phase: Callable[[], bool],
    ) -> None:
        """Initialize the home inspection effect adapter and its owned state."""
        self._view = view
        self._services = services
        self._flow = flow
        self._is_critical_phase = is_critical_phase
        self._generation = 0
        self._active = True

    def activate(self) -> None:
        """Activate the operation owned by this home inspection effect adapter."""
        if self._active:
            return
        self._generation += 1
        self._active = True

    def close(self) -> None:
        """Close resources owned by this home inspection effect adapter."""
        self._generation += 1
        self._active = False

    def invalidate(self) -> None:
        """Invalidate the operation owned by this home inspection effect adapter."""
        self._generation += 1

    def _begin(self) -> int:
        """Begin the operation owned by this home inspection effect adapter."""
        self._generation += 1
        return self._generation

    def _is_current(self, generation: int, owner: ModelAdministrationOwner) -> bool:
        """Return whether current applies to this home inspection effect adapter."""
        return generation == self._generation and self._effects_allowed(owner)

    def _effects_allowed(self, owner: ModelAdministrationOwner) -> bool:
        """Return the effects allowed value owned by this home inspection effect adapter."""
        return bool(
            self._active
            and self._view.inspection_effects_available()
            and not self._is_critical_phase()
            and self._services.administration.models.is_owner_current(owner)
        )

    def _consume(self, generation: int, owner: ModelAdministrationOwner) -> bool:
        """Consume one modal callback before executing its UI effects."""

        if generation != self._generation:
            return False
        is_current = self._is_current(generation, owner)
        self._generation += 1
        return is_current

    def open_configuration(
        self,
        *,
        mode: RLModelRunningModes,
        framework_uuid: str | None,
        selected_session: str | None,
        selected_checkpoint: str | None = None,
    ) -> None:
        """Open the configuration workflow owned by this home inspection effect adapter."""
        plan = self._flow.build_configuration_inspect_plan(
            is_critical_phase=self._is_critical_phase(),
            mode=mode,
            framework_uuid=framework_uuid,
            selected_session=selected_session,
            selected_checkpoint=selected_checkpoint,
        )
        if plan.notification is not None:
            self._view.notify_inspection(plan.notification)
            return
        owner = plan.owner
        target = plan.target
        if owner is None or target is None:
            return
        generation = self._begin()
        if not self._is_current(generation, owner):
            return
        self._start_target(owner, generation, target)

    def open_pack(self, *, reopen_startup_dialog: bool = False) -> None:
        """Open the pack workflow owned by this home inspection effect adapter."""
        context = self._flow.build_pack_selector_context(
            is_critical_phase=self._is_critical_phase(),
            reopen_startup_dialog=reopen_startup_dialog,
        )
        if context.notification is not None:
            self._view.notify_inspection(context.notification)
            if context.reopen_startup_dialog:
                self._view.reopen_inspection_configuration()
            return
        owner = context.owner
        if owner is None:
            return
        generation = self._begin()

        def on_result(result: TensorboardPackDialogResult | None) -> None:
            """Handle the result event for this home inspection effect adapter."""
            if not self._consume(generation, owner):
                return
            self._handle_pack_result(owner, result, reopen_startup_dialog)

        dialog = TensorboardPackDialog()
        if self._is_current(generation, owner):
            self._view.push_tensorboard_pack_dialog(dialog, on_result)

    def _handle_pack_result(
        self,
        owner: ModelAdministrationOwner,
        result: TensorboardPackDialogResult | None,
        reopen_startup_dialog: bool,
    ) -> None:
        """Handle the pack result workflow for this home inspection effect adapter."""
        selection = (
            None
            if result is None
            else TensorboardPackSelection(
                pack_type=result.pack_type,
                framework_uuid=result.framework_uuid,
                run_name=result.run_name,
                logdir=result.logdir,
                configuration=(
                    result.model_manifest.get_configuration()
                    if result.model_manifest is not None
                    else None
                ),
            )
        )
        plan = self._flow.build_pack_open_plan(
            result=selection,
            reopen_startup_dialog=reopen_startup_dialog,
        )
        if self._effects_allowed(owner):
            self._apply_pack_plan(owner, plan)

    def _apply_pack_plan(
        self,
        owner: ModelAdministrationOwner,
        plan: TensorboardPackOpenPlan,
    ) -> None:
        """Apply the pack plan transition owned by this home inspection effect adapter."""
        if plan.canceled:
            if plan.reopen_startup_dialog:
                self._view.reopen_inspection_configuration()
            return
        target = plan.target
        if target is None:
            if plan.notification is not None:
                self._view.notify_inspection(plan.notification)
            return
        generation = self._begin()
        self._start_target(owner, generation, target)

    def _start_target(
        self,
        owner: ModelAdministrationOwner,
        generation: int,
        target: InspectionTarget,
    ) -> None:
        """Schedule one guarded target and apply completion on the UI thread."""

        self._view.begin_control_panel_inspection(
            mode=target.mode,
            framework_uuid=target.framework_uuid,
        )
        self._view.refresh_inspection_runtime()

        def _completed(status: InspectionStatus) -> None:
            if not self._is_current(generation, owner):
                return
            self._view.refresh_inspection_runtime()
            if not self._is_current(generation, owner):
                return
            self._view.notify_inspection(self._flow.build_started_notification(status))

        self._view.start_inspection(target, _completed)
