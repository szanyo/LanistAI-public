# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate home inspect flow."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.gui.contracts import UiNotification
from lanista.inspection_lifecycle import InspectionLifecycleState, InspectionStatus
from lanista.reinforcengine.application_errors import ApplicationError
from lanista.reinforcengine.rl_model.administration_contracts import ModelAdministrationOwner
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.inspection_target_service import InspectionTarget
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.selection_service import TensorboardRestartContext


@dataclass(frozen=True)
class ConfigurationInspectPlan:
    """Describe the executable plan for configuration inspect."""

    notification: UiNotification | None = None
    owner: ModelAdministrationOwner | None = None
    target: InspectionTarget | None = None

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this configuration inspect plan."""
        return self.notification is not None or self.target is None

    @property
    def restart_context(self) -> TensorboardRestartContext | None:
        """Project the legacy native-viewer context from the typed target."""

        if self.target is None:
            return None
        return TensorboardRestartContext(
            mode=self.target.mode,
            framework_uuid=self.target.framework_uuid,
            selected_session=self.target.selected_session,
        )


@dataclass(frozen=True)
class TensorboardPackSelectorContext:
    """Carry the runtime context for tensorboard pack selector."""

    notification: UiNotification | None = None
    owner: ModelAdministrationOwner | None = None
    reopen_startup_dialog: bool = False

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this tensorboard pack selector context."""
        return self.notification is not None


@dataclass(frozen=True, slots=True)
class TensorboardPackSelection:
    """Typed projection of a TensorBoard pack dialog result."""

    pack_type: str
    framework_uuid: str
    run_name: str
    logdir: str
    configuration: object | None = None


@dataclass(frozen=True)
class TensorboardPackOpenPlan:
    """Describe the executable plan for tensorboard pack open."""

    canceled: bool = False
    reopen_startup_dialog: bool = False
    target: InspectionTarget | None = None
    notification: UiNotification | None = None

    @property
    def open_required(self) -> bool:
        """Return the open required exposed by this tensorboard pack open plan."""
        return self.target is not None

    @property
    def logdir(self) -> str | None:
        """Return the native pack directory for compatibility projections."""

        return self.target.native_logdir if self.target is not None else None

    @property
    def online_context(self) -> str | None:
        """Return the selected pack's compact logical context."""

        if self.target is None:
            return None
        return f"{self.target.pack_type}/{self.target.pack_name}"

    @property
    def mode(self) -> RLModelRunningModes | None:
        """Return the target mode."""

        return self.target.mode if self.target is not None else None

    @property
    def framework_uuid(self) -> str | None:
        """Return the target framework."""

        return self.target.framework_uuid if self.target is not None else None


class HomeInspectFlowService:
    """Decision helpers for HomeScreen TensorBoard inspect orchestration."""

    def __init__(self, services: ApplicationServices) -> None:
        """Initialize the home inspect flow service and its owned state."""
        self._services = services

    def _capture_owner(
        self,
        *,
        blocked_title: str,
    ) -> tuple[ModelAdministrationOwner | None, UiNotification | None]:
        """Capture the owner required by this home inspect flow service."""
        try:
            return self._services.administration.models.capture_owner(), None
        except ApplicationError as error:
            return (
                None,
                UiNotification(
                    message=str(error),
                    title=blocked_title,
                    severity="error",
                ),
            )

    def build_configuration_inspect_plan(
        self,
        *,
        is_critical_phase: bool,
        mode: RLModelRunningModes,
        framework_uuid: str | None,
        selected_session: str | None,
        selected_checkpoint: str | None = None,
    ) -> ConfigurationInspectPlan:
        """Build the configuration inspect plan required by this home inspect flow service."""
        blocked_notification = self._build_inspect_block_notification(
            is_critical_phase=is_critical_phase,
        )
        if blocked_notification is not None:
            return ConfigurationInspectPlan(notification=blocked_notification)

        owner, owner_blocker = self._capture_owner(
            blocked_title="Inspect blocked",
        )
        if owner_blocker is not None:
            return ConfigurationInspectPlan(notification=owner_blocker)
        assert owner is not None

        configuration = self._services.runtime.context.get_active_configuration()
        restart_context = self._services.runtime.selection.build_tensorboard_restart_context(
            mode=mode,
            selected_session=selected_session,
            framework_uuid=framework_uuid,
            configuration=configuration,
            current_runtime_framework_uuid=(
                self._services.runtime.context.get_current_framework_uuid()
            ),
        )
        if restart_context is None:
            return ConfigurationInspectPlan(
                notification=UiNotification(
                    message="TensorBoard inspect requires a valid framework selection.",
                    title="Inspect blocked",
                    severity="warning",
                )
            )

        try:
            target = self._services.inspection.targets.ordinary(
                configuration=configuration,
                mode=restart_context.mode,
                framework_uuid=restart_context.framework_uuid,
                selected_session=restart_context.selected_session,
                selected_checkpoint=selected_checkpoint,
            )
        except ApplicationError as error:
            return ConfigurationInspectPlan(
                notification=UiNotification(
                    message=str(error),
                    title="Inspect blocked",
                    severity="warning",
                )
            )
        return ConfigurationInspectPlan(owner=owner, target=target)

    def build_pack_selector_context(
        self,
        *,
        is_critical_phase: bool,
        reopen_startup_dialog: bool,
    ) -> TensorboardPackSelectorContext:
        """Build the pack selector context required by this home inspect flow service."""
        blocked_notification = self._build_inspect_block_notification(
            is_critical_phase=is_critical_phase,
        )
        if blocked_notification is not None:
            return TensorboardPackSelectorContext(
                notification=blocked_notification,
                reopen_startup_dialog=reopen_startup_dialog,
            )

        owner, owner_blocker = self._capture_owner(
            blocked_title="Inspect blocked",
        )
        if owner_blocker is not None:
            return TensorboardPackSelectorContext(
                notification=owner_blocker,
                reopen_startup_dialog=reopen_startup_dialog,
            )
        assert owner is not None
        return TensorboardPackSelectorContext(
            notification=None,
            owner=owner,
            reopen_startup_dialog=reopen_startup_dialog,
        )

    def build_pack_open_plan(
        self,
        *,
        result: TensorboardPackSelection | None,
        reopen_startup_dialog: bool,
    ) -> TensorboardPackOpenPlan:
        """Build the pack open plan required by this home inspect flow service."""
        if result is None:
            return TensorboardPackOpenPlan(
                canceled=True,
                reopen_startup_dialog=reopen_startup_dialog,
            )

        try:
            target = self._services.inspection.targets.pack(
                configuration=(
                    result.configuration
                    or self._services.runtime.context.get_active_configuration()
                ),
                pack_type=result.pack_type,
                framework_uuid=result.framework_uuid,
                pack_name=result.run_name,
                logdir=result.logdir,
            )
        except ApplicationError as error:
            return TensorboardPackOpenPlan(
                notification=UiNotification(
                    message=str(error),
                    title="Inspect blocked",
                    severity="warning",
                )
            )
        return TensorboardPackOpenPlan(
            target=target,
            notification=UiNotification(
                message=f"Inspection target selected: {target.pack_name}",
                title="Inspection",
                severity="information",
            ),
        )

    def build_configuration_opened_notification(
        self,
        *,
        mode: RLModelRunningModes,
        framework_uuid: str,
    ) -> UiNotification:
        """Build the configuration opened notification required by this home inspect flow service.
        """
        return UiNotification(
            message=f"Inspection selected for {mode.value} / {framework_uuid}.",
            title="Inspection",
            severity="information",
        )

    @staticmethod
    def build_started_notification(status: InspectionStatus) -> UiNotification:
        """Describe full, partial, or failed dual-viewer startup truthfully."""

        ready = [
            name
            for name, viewer in (
                ("native TensorBoard", status.native),
                ("Analysis Dashboard", status.analysis),
            )
            if viewer.state is InspectionLifecycleState.READY
        ]
        failed = [
            name
            for name, viewer in (
                ("native TensorBoard", status.native),
                ("Analysis Dashboard", status.analysis),
            )
            if viewer.state is InspectionLifecycleState.FAILED
        ]
        if len(ready) == 2:
            return UiNotification(
                "Native TensorBoard and Analysis Dashboard are ready.",
                "Inspection",
                "information",
            )
        if ready:
            return UiNotification(
                f"Ready: {', '.join(ready)}. Failed: {', '.join(failed)}.",
                "Inspection partially available",
                "warning",
            )
        return UiNotification(
            "Neither inspection viewer could be started.",
            "Inspection failed",
            "error",
        )

    def _build_inspect_block_notification(
        self,
        *,
        is_critical_phase: bool,
    ) -> UiNotification | None:
        """Build the inspect block notification required by this home inspect flow service."""
        if is_critical_phase:
            return UiNotification(
                message="Inspect is disabled while training is running.",
                title="Inspect blocked",
                severity="warning",
            )

        if self._services.runtime.staging.get_selected_manifest() is None:
            return UiNotification(
                message="TensorBoard inspect requires a selected model.",
                title="Inspect blocked",
                severity="warning",
            )

        if not self._services.runtime.status.current_model_has_valid_structure():
            return UiNotification(
                message="TensorBoard inspect requires a structurally valid selected model.",
                title="Inspect blocked",
                severity="warning",
            )

        return None
