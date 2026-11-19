# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Textual boundary for the immutable Home edit workflow."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from textual.screen import Screen

from lanista.gui.contracts import UiNotification
from lanista.gui.predefs.secure_delete_dialog_definitions import secure_delete_model_dialog
from lanista.gui.screens.load_model_dialog import LoadModelDialog, LoadModelDialogResult
from lanista.gui.services.home.home_edit_flow_service import HomeEditFlowService
from lanista.gui.services.home.home_edit_workflow import (
    ApplyDeletedModel,
    ApplySavedDetail,
    CheckpointSelected,
    CommitTarget,
    DeleteConfirmed,
    DeleteFinished,
    DeleteModel,
    DetailResolved,
    DetailSubmitted,
    HomeEditDetailAction,
    HomeEditDetailSubmission,
    HomeEditEffect,
    HomeEditEvent,
    HomeEditPhase,
    HomeEditSelectionAction,
    HomeEditState,
    HomeEditTargetAction,
    ManifestStageFinished,
    ModelSelected,
    Notify,
    OpenCheckpointSelector,
    OpenDeleteConfirmation,
    OpenDetail,
    OpenModelSelector,
    OpenSessionSelector,
    OpenTargetSelector,
    RefreshControlPanel,
    ReopenParent,
    ResolveDetail,
    ResolveSelector,
    ResolveTarget,
    SaveDetail,
    SaveFinished,
    SelectorResolved,
    SessionSelected,
    StageManifest,
    TargetCommitFinished,
    TargetResolved,
    TargetSelected,
    WorkflowFailed,
    reduce_home_edit,
    start_home_edit,
)
from lanista.gui.utils.navigation import route_ids as nav_routes
from lanista.gui.utils.navigation.coordinator import NavigationCoordinator
from lanista.gui.widgets.dialogs.checkpoint_configuration_dialog import (
    CheckpointConfigurationDialog,
    CheckpointConfigurationDialogResult,
)
from lanista.gui.widgets.dialogs.edit_target_selection_dialog import (
    EditTargetSelectionDialog,
    EditTargetSelectionDialogResult,
)
from lanista.gui.widgets.dialogs.model_detail_edit_dialog import (
    ModelDetailEditDialog,
    ModelDetailEditDialogResult,
)
from lanista.gui.widgets.dialogs.session_configuration_dialog import (
    SessionConfigurationDialog,
    SessionConfigurationDialogResult,
)
from lanista.gui.widgets.fragments.control_panel import ControlPanel
from lanista.gui.widgets.fragments.info_panel import InfoPanel
from lanista.reinforcengine.rl_model.application_services import ApplicationServices


@dataclass(frozen=True, slots=True)
class _HomeEditLaunch:
    """Carry the data required for home edit launch."""

    workflow_id: str
    reopen_route: str | None
    reopen_opener: Callable[[], None] | None


class HomeEditWorkflowAdapter:
    """Execute reducer effects at the Textual and application-service boundary."""

    def __init__(
        self,
        *,
        screen: Screen[None],
        services: ApplicationServices,
        flow_service: HomeEditFlowService,
        navigation: NavigationCoordinator,
        is_critical_phase: Callable[[], bool],
        stop_tensorboard: Callable[[], None],
        notify_ui: Callable[[UiNotification], None],
    ) -> None:
        """Initialize the home edit workflow adapter and its owned state."""
        self._screen = screen
        self._services = services
        self._flow_service = flow_service
        self._navigation = navigation
        self._is_critical_phase = is_critical_phase
        self._stop_tensorboard = stop_tensorboard
        self._notify_ui = notify_ui
        self._logger = logging.getLogger(f"{__name__}.{type(screen).__name__}")
        self._state: HomeEditState | None = None
        self._launch: _HomeEditLaunch | None = None
        self._events: deque[HomeEditEvent] = deque()
        self._dispatching = False
        self._effect_handlers: dict[
            type[HomeEditEffect],
            Callable[[HomeEditEffect], None],
        ] = {
            ResolveTarget: self._resolve_target,
            ResolveSelector: self._resolve_selector,
            OpenTargetSelector: self._open_target_selector,
            CommitTarget: self._commit_target,
            OpenModelSelector: self._open_model_selector,
            StageManifest: self._stage_manifest,
            OpenSessionSelector: self._open_session_selector,
            OpenCheckpointSelector: self._open_checkpoint_selector,
            ResolveDetail: self._resolve_detail,
            OpenDetail: self._open_detail,
            OpenDeleteConfirmation: self._open_delete_confirmation,
            SaveDetail: self._save_detail,
            DeleteModel: self._delete_model,
            RefreshControlPanel: self._refresh_control_panel,
            ApplySavedDetail: self._apply_saved_detail,
            ApplyDeletedModel: self._apply_deleted_model,
            Notify: self._notify,
            ReopenParent: self._reopen_parent,
        }

    @property
    def state(self) -> HomeEditState | None:
        """Expose the immutable snapshot for diagnostics and focused UI tests."""

        return self._state

    def start(
        self,
        *,
        reopen_route: str | None,
        reopen_opener: Callable[[], None] | None,
        open_details_directly: bool,
        selected_session_name: str | None,
        selected_checkpoint_name: str | None,
        expected_model_uuid: str | None = None,
    ) -> str:
        """Supersede any older modal chain and start one guarded workflow."""

        workflow_id = str(uuid4())
        transition = start_home_edit(
            workflow_id,
            open_details_directly=open_details_directly,
            selected_session_name=selected_session_name,
            selected_checkpoint_name=selected_checkpoint_name,
            expected_model_uuid=expected_model_uuid,
        )
        self._events.clear()
        self._state = transition.state
        self._launch = _HomeEditLaunch(
            workflow_id,
            reopen_route,
            reopen_opener,
        )
        for effect in transition.effects:
            self._run_effect(effect)
        return workflow_id

    def dispatch(self, event: HomeEditEvent) -> None:
        """Queue one callback result; stale workflow IDs are side-effect free."""

        if not self._is_current(event.workflow_id):
            return
        self._events.append(event)
        if self._dispatching:
            return
        self._dispatching = True
        try:
            self._drain_events()
        finally:
            self._dispatching = False

    def _drain_events(self) -> None:
        """Handle the drain events operation owned by this home edit workflow adapter."""
        while self._events:
            event = self._events.popleft()
            state = self._state
            if state is None or event.workflow_id != state.workflow_id:
                continue
            transition = reduce_home_edit(state, event)
            self._state = transition.state
            for effect in transition.effects:
                self._run_effect(effect)

    def _run_effect(self, effect: HomeEditEffect) -> None:
        """Run the effect owned by this home edit workflow adapter."""
        if not self._is_current(effect.workflow_id):
            return
        handler = self._effect_handlers.get(type(effect))
        if handler is None:
            self._logger.error("Unsupported Home edit effect: %s", type(effect).__name__)
            return
        try:
            handler(effect)
        except Exception as exc:
            self._logger.exception(
                "Unhandled Home edit effect failure: %s",
                type(effect).__name__,
                exc_info=exc,
            )
            if isinstance(effect, Notify):
                return
            notification = UiNotification(
                message="The edit workflow could not be completed.",
                title="Edit failed",
                severity="error",
            )
            state = self._current_state(effect.workflow_id)
            if state is not None and state.phase is not HomeEditPhase.FINISHED:
                self.dispatch(WorkflowFailed(effect.workflow_id, notification))
                return
            self._notify_ui(notification)

    def _is_current(self, workflow_id: str) -> bool:
        """Return whether current applies to this home edit workflow adapter."""
        return self._state is not None and self._state.workflow_id == workflow_id

    def _current_state(self, workflow_id: str) -> HomeEditState | None:
        """Return the current state value owned by this home edit workflow adapter."""
        if not self._is_current(workflow_id):
            return None
        return self._state

    def _fail(self, workflow_id: str, *, title: str, message: str) -> None:
        """Fail the operation owned by this home edit workflow adapter."""
        self.dispatch(
            WorkflowFailed(
                workflow_id,
                UiNotification(message=message, title=title, severity="error"),
            )
        )

    def _resolve_target(self, raw_effect: HomeEditEffect) -> None:
        """Resolve the target required by this home edit workflow adapter."""
        assert isinstance(raw_effect, ResolveTarget)
        state = self._current_state(raw_effect.workflow_id)
        if state is None:
            return
        try:
            control_panel = self._screen.query_one("#control_panel", ControlPanel)
            context = self._flow_service.build_target_context(
                is_critical_phase=self._is_critical_phase(),
                selected_session_name=state.selected_session_name,
                selected_checkpoint_name=state.selected_checkpoint_name,
                control_panel_session_name=control_panel.get_selected_session_name(),
                expected_model_uuid=state.expected_model_uuid,
            )
        except Exception as exc:
            self._logger.exception("Failed to resolve Home edit target", exc_info=exc)
            self._fail(
                raw_effect.workflow_id,
                title="Edit unavailable",
                message="Failed to resolve editable model metadata.",
            )
            return
        self.dispatch(TargetResolved(raw_effect.workflow_id, context))

    def _resolve_selector(self, raw_effect: HomeEditEffect) -> None:
        """Resolve the selector required by this home edit workflow adapter."""
        assert isinstance(raw_effect, ResolveSelector)
        state = self._current_state(raw_effect.workflow_id)
        if state is None or state.payload is None:
            self._fail_missing_context(raw_effect.workflow_id)
            return
        try:
            context = self._flow_service.build_target_selector_context(
                payload=state.payload,
                preferred_session_name=state.target_session_name,
                preferred_checkpoint_name=state.target_checkpoint_name,
                fallback_session_name=state.target_session_name,
            )
        except Exception as exc:
            self._logger.exception("Failed to resolve Home edit selector", exc_info=exc)
            self._fail_missing_context(raw_effect.workflow_id)
            return
        self.dispatch(SelectorResolved(raw_effect.workflow_id, context))

    def _fail_missing_context(self, workflow_id: str) -> None:
        """Fail the missing context owned by this home edit workflow
adapter."""
        self._fail(
            workflow_id,
            title="Edit unavailable",
            message="The selected model context is no longer available.",
        )

    def _open_target_selector(self, raw_effect: HomeEditEffect) -> None:
        """Open the target selector workflow owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, OpenTargetSelector)
        context = raw_effect.context

        def selected(result: EditTargetSelectionDialogResult | None) -> None:
            """Perform the selected step within the enclosing open target selector workflow."""
            self.dispatch(
                TargetSelected(
                    raw_effect.workflow_id,
                    self._target_action(result.action if result is not None else None),
                    result.session_name if result is not None else None,
                    result.checkpoint_name if result is not None else None,
                )
            )

        self._screen.app.push_screen(
            EditTargetSelectionDialog(
                loaded_model_name=context.loaded_model_name,
                model_uuid=context.model_uuid,
                session_options=list(context.session_options),
                selected_session_name=context.selected_session_name,
                checkpoint_options=list(context.checkpoint_options),
                selected_checkpoint_name=context.selected_checkpoint_name,
            ),
            selected,
        )

    def _commit_target(self, raw_effect: HomeEditEffect) -> None:
        """Commit the target owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, CommitTarget)
        if not self._is_current(raw_effect.workflow_id):
            return
        result = self._flow_service.select_edit_target(
            session_name=raw_effect.session_name,
            checkpoint_name=raw_effect.checkpoint_name,
            expected_model_uuid=raw_effect.expected_model_uuid,
        )
        if not result.succeeded or result.value is None:
            failure_detail = result.failure
            self._logger.error(
                "Home edit target selection failed; operation_id=%s; code=%s.",
                result.operation_id,
                (failure_detail.code.value if failure_detail is not None else "missing_result"),
            )
            failure = UiNotification(
                message=(
                    failure_detail.message
                    if failure_detail is not None
                    else "Failed to save the selected edit target."
                ),
                title="Selection failed",
                severity="error",
            )
            self.dispatch(TargetCommitFinished(raw_effect.workflow_id, False, failure))
            return
        self.dispatch(TargetCommitFinished(raw_effect.workflow_id, True))

    def _open_model_selector(self, raw_effect: HomeEditEffect) -> None:
        """Open the model selector workflow owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, OpenModelSelector)
        current_manifest = self._services.runtime.staging.get_selected_manifest()
        current_model_uuid = ""
        if current_manifest is not None:
            configuration = current_manifest.get_configuration()
            current_model_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()

        def selected(result: LoadModelDialogResult | None) -> None:
            """Perform the selected step within the enclosing open model selector workflow."""
            manifest = result.model_manifest if result is not None else None
            self.dispatch(ModelSelected(raw_effect.workflow_id, manifest))

        self._screen.app.push_screen(
            LoadModelDialog(current_model_uuid=current_model_uuid),
            selected,
        )

    def _stage_manifest(self, raw_effect: HomeEditEffect) -> None:
        """Stage the manifest owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, StageManifest)
        if not self._is_current(raw_effect.workflow_id):
            return
        try:
            self._services.runtime.staging.stage_manifest(raw_effect.manifest)
        except Exception as exc:
            self._logger.exception("Failed to stage selected model", exc_info=exc)
            failure = UiNotification(
                message="Failed to select the requested model.",
                title="Model selection failed",
                severity="error",
            )
            self.dispatch(ManifestStageFinished(raw_effect.workflow_id, False, failure))
            return
        self.dispatch(ManifestStageFinished(raw_effect.workflow_id, True))

    def _open_session_selector(self, raw_effect: HomeEditEffect) -> None:
        """Open the session selector workflow owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, OpenSessionSelector)
        context = raw_effect.context

        def selected(result: SessionConfigurationDialogResult | None) -> None:
            """Perform the selected step within the enclosing open session selector workflow."""
            self.dispatch(
                SessionSelected(
                    raw_effect.workflow_id,
                    self._selection_action(result.action if result is not None else None),
                    result.session_name if result is not None else None,
                )
            )

        self._screen.app.push_screen(
            SessionConfigurationDialog(
                session_options=list(context.session_options),
                selected_session_name=context.selected_session_name,
                allow_delete=False,
                allow_edit=False,
            ),
            selected,
        )

    def _open_checkpoint_selector(self, raw_effect: HomeEditEffect) -> None:
        """Open the checkpoint selector workflow owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, OpenCheckpointSelector)
        state = self._current_state(raw_effect.workflow_id)
        if state is None:
            return
        context = raw_effect.context

        def selected(result: CheckpointConfigurationDialogResult | None) -> None:
            """Perform the selected step within the enclosing open checkpoint selector workflow."""
            self.dispatch(
                CheckpointSelected(
                    raw_effect.workflow_id,
                    self._selection_action(result.action if result is not None else None),
                    result.checkpoint_name if result is not None else None,
                )
            )

        self._screen.app.push_screen(
            CheckpointConfigurationDialog(
                session_name=state.target_session_name,
                checkpoint_options=list(context.checkpoint_options),
                selected_checkpoint_name=state.target_checkpoint_name,
                allow_fresh_training=True,
                allow_delete=False,
                allow_rename=False,
            ),
            selected,
        )

    def _resolve_detail(self, raw_effect: HomeEditEffect) -> None:
        """Resolve the detail required by this home edit workflow adapter."""
        assert isinstance(raw_effect, ResolveDetail)
        if raw_effect.deferred:
            scheduled = self._navigation.reopen_parent_later(
                nav_routes.SECURE_DELETE_DIALOG,
                opener=lambda: self._resolve_detail_now(raw_effect),
                guard=lambda: self._is_current(raw_effect.workflow_id),
            )
            if scheduled:
                return
        self._resolve_detail_now(raw_effect)

    def _resolve_detail_now(self, effect: ResolveDetail) -> None:
        """Resolve the detail now required by this home edit workflow adapter."""
        state = self._current_state(effect.workflow_id)
        if state is None:
            return
        if state.payload is None:
            self._fail_missing_context(effect.workflow_id)
            return
        try:
            context = self._flow_service.build_detail_dialog_context(
                payload=state.payload,
                target_checkpoint_name=state.target_checkpoint_name,
            )
        except Exception as exc:
            self._logger.exception("Failed to capture Home edit snapshot", exc_info=exc)
            self._fail_missing_context(effect.workflow_id)
            return
        self.dispatch(DetailResolved(effect.workflow_id, context))

    def _open_detail(self, raw_effect: HomeEditEffect) -> None:
        """Open the detail workflow owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, OpenDetail)
        context = raw_effect.context

        def submitted(result: ModelDetailEditDialogResult | None) -> None:
            """Perform the submitted step within the enclosing open detail workflow."""
            submission = self._detail_submission(result)
            self.dispatch(DetailSubmitted(raw_effect.workflow_id, submission))

        self._screen.app.push_screen(
            ModelDetailEditDialog(
                model_name=context.model_name,
                model_uuid=context.model_uuid,
                model_version=context.model_version,
                framework_uuid=context.framework_uuid,
                model_description=context.model_description,
                session_uuid=context.session_uuid,
                session_name=context.session_name,
                session_description=context.session_description,
                checkpoint_name=context.checkpoint_name,
                checkpoint_annotation=context.checkpoint_annotation,
            ),
            submitted,
        )

    @staticmethod
    def _detail_submission(
        result: ModelDetailEditDialogResult | None,
    ) -> HomeEditDetailSubmission | None:
        """Return the detail submission value owned by this home edit workflow adapter."""
        if result is None:
            return None
        try:
            action = HomeEditDetailAction(result.action)
        except (TypeError, ValueError):
            return None
        return HomeEditDetailSubmission(
            action=action,
            model_name=result.model_name,
            model_version=result.model_version,
            model_description=result.model_description,
            session_uuid=result.session_uuid,
            session_name=result.session_name,
            session_description=result.session_description,
            checkpoint_name=result.checkpoint_name,
            checkpoint_annotation=result.checkpoint_annotation,
        )

    @staticmethod
    def _target_action(value: str | None) -> HomeEditTargetAction | None:
        """Return the target action value owned by this home edit workflow adapter."""
        if value is None:
            return None
        try:
            return HomeEditTargetAction(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _selection_action(value: str | None) -> HomeEditSelectionAction | None:
        """Return the selection action value owned by this home edit workflow adapter."""
        if value is None:
            return None
        try:
            return HomeEditSelectionAction(value)
        except (TypeError, ValueError):
            return None

    def _open_delete_confirmation(self, raw_effect: HomeEditEffect) -> None:
        """Open the delete confirmation workflow owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, OpenDeleteConfirmation)

        def confirmed(result: bool | None) -> None:
            """Perform the confirmed step within the enclosing open delete confirmation workflow."""
            self.dispatch(DeleteConfirmed(raw_effect.workflow_id, bool(result)))

        self._screen.app.push_screen(
            secure_delete_model_dialog(raw_effect.model_name),
            confirmed,
        )

    def _save_detail(self, raw_effect: HomeEditEffect) -> None:
        """Persist the detail owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, SaveDetail)
        if not self._is_current(raw_effect.workflow_id):
            return
        submission = raw_effect.submission
        try:
            result = self._flow_service.save_detail_updates(
                payload=raw_effect.payload,
                model_name=submission.model_name,
                model_version=submission.model_version,
                model_description=submission.model_description,
                session_uuid=submission.session_uuid,
                session_name=submission.session_name,
                session_description=submission.session_description,
                checkpoint_name=submission.checkpoint_name,
                checkpoint_annotation=submission.checkpoint_annotation,
                snapshot=raw_effect.snapshot,
            )
        except Exception as exc:
            self._logger.exception("Failed to save Home edit metadata", exc_info=exc)
            self._fail(
                raw_effect.workflow_id,
                title="Save failed",
                message="Failed to save metadata changes.",
            )
            return
        self.dispatch(SaveFinished(raw_effect.workflow_id, result))

    def _delete_model(self, raw_effect: HomeEditEffect) -> None:
        """Delete the model through this home edit workflow adapter's safety boundary."""
        assert isinstance(raw_effect, DeleteModel)
        if not self._is_current(raw_effect.workflow_id):
            return
        result = self._services.administration.models.delete_current_model_storage(
            expected_model_uuid=raw_effect.expected_model_uuid
        )
        if result.failure is not None:
            self._logger.warning(
                "Model deletion failed "
                f"operation_id={result.operation_id} "
                f"failure_code={result.failure.code}"
            )
        outcome = result.value
        self.dispatch(
            DeleteFinished(
                raw_effect.workflow_id,
                result.succeeded,
                degraded=bool(outcome is not None and outcome.degraded),
            )
        )

    def _refresh_control_panel(self, raw_effect: HomeEditEffect) -> None:
        """Refresh the control panel owned by this home edit workflow
adapter."""
        assert isinstance(raw_effect, RefreshControlPanel)
        self._screen.query_one(
            "#control_panel",
            ControlPanel,
        ).refresh_runtime_context_from_model()

    def _apply_saved_detail(self, raw_effect: HomeEditEffect) -> None:
        """Apply the saved detail transition owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, ApplySavedDetail)
        self._screen.query_one("#info_panel", InfoPanel).set_model_info(
            raw_effect.result.model_display_name,
            raw_effect.result.framework_uuid,
        )
        self._screen.query_one(
            "#control_panel",
            ControlPanel,
        ).refresh_runtime_context_from_model()

    def _apply_deleted_model(self, raw_effect: HomeEditEffect) -> None:
        """Apply the deleted model transition owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, ApplyDeletedModel)
        self._stop_tensorboard()
        info_panel = self._screen.query_one("#info_panel", InfoPanel)
        info_panel.set_model_info(None, None)
        info_panel.set_runtime_info("-", None, None, None, "offline", None)
        control_panel = self._screen.query_one("#control_panel", ControlPanel)
        control_panel.refresh_runtime_context_from_model()
        control_panel.set_main_status("Idle", "-", 0, None)

    def _notify(self, raw_effect: HomeEditEffect) -> None:
        """Notify the operation owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, Notify)
        self._notify_ui(raw_effect.notification)

    def _reopen_parent(self, raw_effect: HomeEditEffect) -> None:
        """Reopen the parent owned by this home edit workflow adapter."""
        assert isinstance(raw_effect, ReopenParent)
        launch = self._launch
        if launch is None or launch.workflow_id != raw_effect.workflow_id:
            return
        self._launch = None

        def workflow_is_current() -> bool:
            """Return whether workflow is current applies to this home edit workflow adapter."""
            return self._is_current(raw_effect.workflow_id)

        if launch.reopen_route is not None:
            self._navigation.reopen_route_later(
                launch.reopen_route,
                opener=launch.reopen_opener,
                guard=workflow_is_current,
            )
        elif launch.reopen_opener is not None:
            self._navigation.schedule_later(
                launch.reopen_opener,
                guard=workflow_is_current,
            )
