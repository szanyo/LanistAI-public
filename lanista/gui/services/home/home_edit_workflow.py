# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Pure state transitions for the Home metadata-edit modal workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

from lanista.gui.contracts import UiNotification
from lanista.gui.services.home.home_edit_flow_service import (
    HomeEditDetailDialogContext,
    HomeEditPayload,
    HomeEditSaveResult,
    HomeEditTargetContext,
    HomeEditTargetSelectorContext,
)
from lanista.reinforcengine.rl_model.metadata.edit_service import MetadataEditSnapshot
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest


class HomeEditPhase(StrEnum):
    """Stable phases used to reject out-of-order modal results."""

    RESOLVING_TARGET = "resolving_target"
    RESOLVING_SELECTOR = "resolving_selector"
    SELECTING_TARGET = "selecting_target"
    COMMITTING_TARGET = "committing_target"
    SELECTING_MODEL = "selecting_model"
    STAGING_MODEL = "staging_model"
    SELECTING_SESSION = "selecting_session"
    SELECTING_CHECKPOINT = "selecting_checkpoint"
    RESOLVING_DETAIL = "resolving_detail"
    EDITING_DETAIL = "editing_detail"
    CONFIRMING_DELETE = "confirming_delete"
    SAVING = "saving"
    DELETING = "deleting"
    FINISHED = "finished"


class HomeEditTargetAction(StrEnum):
    """Normalized actions accepted from the edit-target selector."""

    EDIT = "edit"
    LOAD_MODEL = "load_model"
    SELECT_SESSION = "select_session"
    SELECT_CHECKPOINT = "select_checkpoint"


class HomeEditSelectionAction(StrEnum):
    """Normalized selection-dialog actions."""

    SELECT = "select"


class HomeEditDetailAction(StrEnum):
    """Mutating actions accepted from the model-detail dialog."""

    SAVE = "save"
    DELETE_MODEL = "delete_model"


@dataclass(frozen=True, slots=True)
class HomeEditState:
    """Immutable source of truth for one edit workflow instance."""

    workflow_id: str
    phase: HomeEditPhase
    open_details_directly: bool = False
    selected_session_name: str | None = None
    selected_checkpoint_name: str | None = None
    expected_model_uuid: str | None = None
    target_session_name: str | None = None
    target_checkpoint_name: str | None = None
    payload: HomeEditPayload | None = None
    selector_context: HomeEditTargetSelectorContext | None = None
    detail_context: HomeEditDetailDialogContext | None = None
    pending_delete_model_name: str | None = None


@dataclass(frozen=True, slots=True)
class HomeEditDetailSubmission:
    """Textual-independent values returned by the detail dialog."""

    action: HomeEditDetailAction
    model_name: str | None
    model_version: str | None
    model_description: str | None
    session_uuid: str | None = None
    session_name: str | None = None
    session_description: str | None = None
    checkpoint_name: str | None = None
    checkpoint_annotation: str | None = None


@dataclass(frozen=True, slots=True)
class HomeEditEvent:
    """Represent the home edit lifecycle event."""

    workflow_id: str


@dataclass(frozen=True, slots=True)
class TargetResolved(HomeEditEvent):
    """Carry target resolved data through the workflow event stream."""

    context: HomeEditTargetContext


@dataclass(frozen=True, slots=True)
class SelectorResolved(HomeEditEvent):
    """Carry selector resolved data through the workflow event stream."""

    context: HomeEditTargetSelectorContext


@dataclass(frozen=True, slots=True)
class TargetSelected(HomeEditEvent):
    """Carry target selected data through the workflow event stream."""

    action: HomeEditTargetAction | None
    session_name: str | None
    checkpoint_name: str | None


@dataclass(frozen=True, slots=True)
class TargetCommitFinished(HomeEditEvent):
    """Carry target commit finished data through the workflow event stream."""

    succeeded: bool
    failure: UiNotification | None = None


@dataclass(frozen=True, slots=True)
class ModelSelected(HomeEditEvent):
    """Carry model selected data through the workflow event stream."""

    manifest: ModelManifest | None


@dataclass(frozen=True, slots=True)
class ManifestStageFinished(HomeEditEvent):
    """Carry manifest stage finished data through the workflow event stream."""

    succeeded: bool
    failure: UiNotification | None = None


@dataclass(frozen=True, slots=True)
class SessionSelected(HomeEditEvent):
    """Carry session selected data through the workflow event stream."""

    action: HomeEditSelectionAction | None
    session_name: str | None


@dataclass(frozen=True, slots=True)
class CheckpointSelected(HomeEditEvent):
    """Carry checkpoint selected data through the workflow event stream."""

    action: HomeEditSelectionAction | None
    checkpoint_name: str | None


@dataclass(frozen=True, slots=True)
class DetailResolved(HomeEditEvent):
    """Carry detail resolved data through the workflow event stream."""

    context: HomeEditDetailDialogContext


@dataclass(frozen=True, slots=True)
class DetailSubmitted(HomeEditEvent):
    """Carry detail submitted data through the workflow event stream."""

    submission: HomeEditDetailSubmission | None


@dataclass(frozen=True, slots=True)
class DeleteConfirmed(HomeEditEvent):
    """Carry delete confirmed data through the workflow event stream."""

    confirmed: bool


@dataclass(frozen=True, slots=True)
class DeleteFinished(HomeEditEvent):
    """Carry delete finished data through the workflow event stream."""

    succeeded: bool
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class SaveFinished(HomeEditEvent):
    """Carry save finished data through the workflow event stream."""

    result: HomeEditSaveResult


@dataclass(frozen=True, slots=True)
class WorkflowFailed(HomeEditEvent):
    """Carry workflow failed data through the workflow event stream."""

    notification: UiNotification


@dataclass(frozen=True, slots=True)
class HomeEditEffect:
    """Represent the home edit workflow operation."""

    workflow_id: str


@dataclass(frozen=True, slots=True)
class ResolveTarget(HomeEditEffect):
    """Describe the resolve target workflow effect."""

    pass


@dataclass(frozen=True, slots=True)
class ResolveSelector(HomeEditEffect):
    """Describe the resolve selector workflow effect."""

    pass


@dataclass(frozen=True, slots=True)
class OpenTargetSelector(HomeEditEffect):
    """Describe the open target selector workflow effect."""

    context: HomeEditTargetSelectorContext


@dataclass(frozen=True, slots=True)
class CommitTarget(HomeEditEffect):
    """Describe the commit target workflow effect."""

    session_name: str | None
    checkpoint_name: str | None
    expected_model_uuid: str


@dataclass(frozen=True, slots=True)
class OpenModelSelector(HomeEditEffect):
    """Describe the open model selector workflow effect."""

    pass


@dataclass(frozen=True, slots=True)
class StageManifest(HomeEditEffect):
    """Describe the stage manifest workflow effect."""

    manifest: ModelManifest


@dataclass(frozen=True, slots=True)
class OpenSessionSelector(HomeEditEffect):
    """Describe the open session selector workflow effect."""

    context: HomeEditTargetSelectorContext


@dataclass(frozen=True, slots=True)
class OpenCheckpointSelector(HomeEditEffect):
    """Describe the open checkpoint selector workflow effect."""

    context: HomeEditTargetSelectorContext


@dataclass(frozen=True, slots=True)
class ResolveDetail(HomeEditEffect):
    """Describe the resolve detail workflow effect."""

    deferred: bool = False


@dataclass(frozen=True, slots=True)
class OpenDetail(HomeEditEffect):
    """Describe the open detail workflow effect."""

    context: HomeEditDetailDialogContext


@dataclass(frozen=True, slots=True)
class OpenDeleteConfirmation(HomeEditEffect):
    """Describe the open delete confirmation workflow effect."""

    model_name: str


@dataclass(frozen=True, slots=True)
class SaveDetail(HomeEditEffect):
    """Describe the save detail workflow effect."""

    payload: HomeEditPayload
    submission: HomeEditDetailSubmission
    snapshot: MetadataEditSnapshot


@dataclass(frozen=True, slots=True)
class DeleteModel(HomeEditEffect):
    """Describe the delete model workflow effect."""

    model_name: str
    expected_model_uuid: str


@dataclass(frozen=True, slots=True)
class RefreshControlPanel(HomeEditEffect):
    """Describe the refresh control panel workflow effect."""

    pass


@dataclass(frozen=True, slots=True)
class ApplySavedDetail(HomeEditEffect):
    """Describe the apply saved detail workflow effect."""

    result: HomeEditSaveResult


@dataclass(frozen=True, slots=True)
class ApplyDeletedModel(HomeEditEffect):
    """Describe the apply deleted model workflow effect."""

    model_name: str


@dataclass(frozen=True, slots=True)
class Notify(HomeEditEffect):
    """Describe the notify workflow effect."""

    notification: UiNotification


@dataclass(frozen=True, slots=True)
class ReopenParent(HomeEditEffect):
    """Describe the reopen parent workflow effect."""

    pass


@dataclass(frozen=True, slots=True)
class HomeEditTransition:
    """Represent the home edit workflow operation."""

    state: HomeEditState
    effects: tuple[HomeEditEffect, ...] = ()


def start_home_edit(
    workflow_id: str,
    *,
    open_details_directly: bool,
    selected_session_name: str | None,
    selected_checkpoint_name: str | None,
    expected_model_uuid: str | None = None,
) -> HomeEditTransition:
    """Create one workflow and request its first domain snapshot."""

    state = HomeEditState(
        workflow_id=workflow_id,
        phase=HomeEditPhase.RESOLVING_TARGET,
        open_details_directly=open_details_directly,
        selected_session_name=selected_session_name,
        selected_checkpoint_name=selected_checkpoint_name,
        expected_model_uuid=expected_model_uuid,
    )
    return HomeEditTransition(state, (ResolveTarget(workflow_id),))


def _unchanged(state: HomeEditState) -> HomeEditTransition:
    """Return the unchanged value owned by this home edit workflow module."""
    return HomeEditTransition(state)


def _finish(
    state: HomeEditState,
    *effects: HomeEditEffect,
) -> HomeEditTransition:
    """Return the finish value owned by this home edit workflow module."""
    return HomeEditTransition(
        replace(state, phase=HomeEditPhase.FINISHED),
        effects,
    )


def _restart_target_resolution(
    state: HomeEditState,
    *,
    open_details_directly: bool,
    selected_session_name: str | None,
    selected_checkpoint_name: str | None,
) -> HomeEditState:
    """Discard stale snapshots before rebuilding the selected model context."""

    return replace(
        state,
        phase=HomeEditPhase.RESOLVING_TARGET,
        open_details_directly=open_details_directly,
        selected_session_name=selected_session_name,
        selected_checkpoint_name=selected_checkpoint_name,
        target_session_name=None,
        target_checkpoint_name=None,
        payload=None,
        selector_context=None,
        detail_context=None,
        pending_delete_model_name=None,
    )


def _target_resolved(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Return the target resolved value owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.RESOLVING_TARGET:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, TargetResolved)
    context = event.context
    if context.blocked_notification is not None:
        return _finish(
            state,
            Notify(state.workflow_id, context.blocked_notification),
            ReopenParent(state.workflow_id),
        )
    if context.payload is None:
        failure = UiNotification(
            message="Failed to resolve editable model metadata.",
            title="Edit unavailable",
            severity="error",
        )
        return _finish(
            state,
            Notify(state.workflow_id, failure),
            ReopenParent(state.workflow_id),
        )
    next_state = replace(
        state,
        payload=context.payload,
        expected_model_uuid=context.payload.model_uuid,
        target_session_name=context.target_session_name,
        target_checkpoint_name=context.target_checkpoint_name,
    )
    if state.open_details_directly:
        next_state = replace(next_state, phase=HomeEditPhase.RESOLVING_DETAIL)
        return HomeEditTransition(
            next_state,
            (ResolveDetail(state.workflow_id),),
        )
    next_state = replace(next_state, phase=HomeEditPhase.RESOLVING_SELECTOR)
    return HomeEditTransition(next_state, (ResolveSelector(state.workflow_id),))


def _selector_resolved(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Return the selector resolved value owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.RESOLVING_SELECTOR:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, SelectorResolved)
    next_state = replace(
        state,
        phase=HomeEditPhase.SELECTING_TARGET,
        selector_context=event.context,
    )
    return HomeEditTransition(
        next_state,
        (OpenTargetSelector(state.workflow_id, event.context),),
    )


def _target_selected(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Return the target selected value owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.SELECTING_TARGET:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, TargetSelected)
    selected_state = replace(
        state,
        target_session_name=event.session_name,
        target_checkpoint_name=event.checkpoint_name,
    )
    if event.action is HomeEditTargetAction.EDIT:
        if state.payload is None:
            failure = UiNotification(
                message="The selected model context is no longer available.",
                title="Selection failed",
                severity="error",
            )
            return _finish(
                state,
                Notify(state.workflow_id, failure),
                ReopenParent(state.workflow_id),
            )
        next_state = replace(
            selected_state,
            phase=HomeEditPhase.COMMITTING_TARGET,
        )
        return HomeEditTransition(
            next_state,
            (
                CommitTarget(
                    state.workflow_id,
                    event.session_name,
                    event.checkpoint_name,
                    state.payload.model_uuid,
                ),
            ),
        )
    if event.action is HomeEditTargetAction.LOAD_MODEL:
        return HomeEditTransition(
            replace(selected_state, phase=HomeEditPhase.SELECTING_MODEL),
            (OpenModelSelector(state.workflow_id),),
        )
    if event.action is HomeEditTargetAction.SELECT_SESSION and state.selector_context is not None:
        return HomeEditTransition(
            replace(selected_state, phase=HomeEditPhase.SELECTING_SESSION),
            (OpenSessionSelector(state.workflow_id, state.selector_context),),
        )
    if (
        event.action is HomeEditTargetAction.SELECT_CHECKPOINT
        and state.selector_context is not None
    ):
        return HomeEditTransition(
            replace(selected_state, phase=HomeEditPhase.SELECTING_CHECKPOINT),
            (OpenCheckpointSelector(state.workflow_id, state.selector_context),),
        )
    return _finish(state, ReopenParent(state.workflow_id))


def _target_commit_finished(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Return the target commit finished value owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.COMMITTING_TARGET:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, TargetCommitFinished)
    if not event.succeeded:
        failure = event.failure or UiNotification(
            message="Failed to save the selected edit target.",
            title="Selection failed",
            severity="error",
        )
        return _finish(
            state,
            Notify(state.workflow_id, failure),
            ReopenParent(state.workflow_id),
        )
    next_state = _restart_target_resolution(
        state,
        open_details_directly=True,
        selected_session_name=state.target_session_name,
        selected_checkpoint_name=state.target_checkpoint_name,
    )
    return HomeEditTransition(
        next_state,
        (
            RefreshControlPanel(state.workflow_id),
            ResolveTarget(state.workflow_id),
        ),
    )


def _model_selected(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Return the model selected value owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.SELECTING_MODEL:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, ModelSelected)
    if event.manifest is None:
        next_state = _restart_target_resolution(
            state,
            open_details_directly=False,
            selected_session_name=None,
            selected_checkpoint_name=None,
        )
        return HomeEditTransition(
            next_state,
            (ResolveTarget(state.workflow_id),),
        )
    return HomeEditTransition(
        replace(state, phase=HomeEditPhase.STAGING_MODEL),
        (StageManifest(state.workflow_id, event.manifest),),
    )


def _manifest_stage_finished(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Return the manifest stage finished value owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.STAGING_MODEL:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, ManifestStageFinished)
    if not event.succeeded:
        failure = event.failure or UiNotification(
            message="Failed to select the requested model.",
            title="Model selection failed",
            severity="error",
        )
        effects: list[HomeEditEffect] = [
            Notify(state.workflow_id, failure),
        ]
        next_state = _restart_target_resolution(
            state,
            open_details_directly=False,
            selected_session_name=None,
            selected_checkpoint_name=None,
        )
        effects.append(ResolveTarget(state.workflow_id))
        return HomeEditTransition(next_state, tuple(effects))
    next_state = _restart_target_resolution(
        state,
        open_details_directly=False,
        selected_session_name=None,
        selected_checkpoint_name=None,
    )
    return HomeEditTransition(
        next_state,
        (
            RefreshControlPanel(state.workflow_id),
            ResolveTarget(state.workflow_id),
        ),
    )


def _session_selected(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Return the session selected value owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.SELECTING_SESSION:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, SessionSelected)
    session_name = state.target_session_name
    if event.action is HomeEditSelectionAction.SELECT:
        session_name = event.session_name or session_name
    next_state = _restart_target_resolution(
        state,
        open_details_directly=False,
        selected_session_name=session_name,
        selected_checkpoint_name=None,
    )
    return HomeEditTransition(next_state, (ResolveTarget(state.workflow_id),))


def _checkpoint_selected(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Checkpoint the selected owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.SELECTING_CHECKPOINT:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, CheckpointSelected)
    checkpoint_name = state.target_checkpoint_name
    if event.action is HomeEditSelectionAction.SELECT:
        checkpoint_name = event.checkpoint_name or checkpoint_name
    next_state = _restart_target_resolution(
        state,
        open_details_directly=False,
        selected_session_name=state.target_session_name,
        selected_checkpoint_name=checkpoint_name,
    )
    return HomeEditTransition(next_state, (ResolveTarget(state.workflow_id),))


def _detail_resolved(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Return the detail resolved value owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.RESOLVING_DETAIL:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, DetailResolved)
    context = event.context
    if context.snapshot is None:
        failure = context.snapshot_failure or UiNotification(
            message="Failed to capture a safe metadata edit snapshot.",
            title="Edit unavailable",
            severity="error",
        )
        return _finish(
            state,
            Notify(state.workflow_id, failure),
            ReopenParent(state.workflow_id),
        )
    next_state = replace(
        state,
        phase=HomeEditPhase.EDITING_DETAIL,
        detail_context=context,
    )
    return HomeEditTransition(
        next_state,
        (OpenDetail(state.workflow_id, context),),
    )


def _detail_submitted(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Return the detail submitted value owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.EDITING_DETAIL:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, DetailSubmitted)
    submission = event.submission
    if submission is None:
        return _finish(state, ReopenParent(state.workflow_id))
    if submission.action is HomeEditDetailAction.DELETE_MODEL:
        fallback_name = getattr(state.payload, "model_name", None)
        model_name = submission.model_name or fallback_name or "current model"
        return HomeEditTransition(
            replace(
                state,
                phase=HomeEditPhase.CONFIRMING_DELETE,
                pending_delete_model_name=model_name,
            ),
            (OpenDeleteConfirmation(state.workflow_id, model_name),),
        )
    detail_context = state.detail_context
    if state.payload is None or detail_context is None or detail_context.snapshot is None:
        return _finish(state, ReopenParent(state.workflow_id))
    return HomeEditTransition(
        replace(state, phase=HomeEditPhase.SAVING),
        (
            SaveDetail(
                state.workflow_id,
                state.payload,
                submission,
                detail_context.snapshot,
            ),
        ),
    )


def _delete_confirmed(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Delete the confirmed through this home edit workflow module's safety boundary."""
    if state.phase is not HomeEditPhase.CONFIRMING_DELETE:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, DeleteConfirmed)
    if not event.confirmed:
        next_state = replace(state, phase=HomeEditPhase.RESOLVING_DETAIL)
        return HomeEditTransition(
            next_state,
            (ResolveDetail(state.workflow_id, deferred=True),),
        )
    detail_context = state.detail_context
    if detail_context is None or detail_context.snapshot is None:
        failure = UiNotification(
            message="The model changed before deletion could be confirmed.",
            title="Delete cancelled",
            severity="warning",
        )
        return _finish(
            state,
            Notify(state.workflow_id, failure),
            ReopenParent(state.workflow_id),
        )
    model_name = state.pending_delete_model_name or "current model"
    return HomeEditTransition(
        replace(state, phase=HomeEditPhase.DELETING),
        (
            DeleteModel(
                state.workflow_id,
                model_name,
                detail_context.snapshot.model_uuid,
            ),
        ),
    )


def _delete_finished(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Delete the finished through this home edit workflow module's safety boundary."""
    if state.phase is not HomeEditPhase.DELETING:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, DeleteFinished)
    if not event.succeeded:
        failure = UiNotification(
            message="Failed to delete model.",
            title="Delete failed",
            severity="error",
        )
        next_state = replace(state, phase=HomeEditPhase.RESOLVING_DETAIL)
        return HomeEditTransition(
            next_state,
            (
                Notify(state.workflow_id, failure),
                ResolveDetail(state.workflow_id, deferred=True),
            ),
        )
    model_name = state.pending_delete_model_name or "current model"
    notification = (
        UiNotification(
            message=(
                f"Model storage was deleted: {model_name}. "
                "Runtime reload or storage recovery is required."
            ),
            title="Model deleted with recovery required",
            severity="warning",
        )
        if event.degraded
        else UiNotification(
            message=f"Model deleted: {model_name}",
            title="Model deleted",
            severity="information",
        )
    )
    return _finish(
        state,
        ApplyDeletedModel(state.workflow_id, model_name),
        Notify(state.workflow_id, notification),
        ReopenParent(state.workflow_id),
    )


def _save_finished(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Persist the finished owned by this home edit workflow module."""
    if state.phase is not HomeEditPhase.SAVING:
        return _unchanged(state)
    event = raw_event
    assert isinstance(event, SaveFinished)
    effects: list[HomeEditEffect] = []
    if event.result.success and not event.result.requires_reload:
        effects.append(ApplySavedDetail(state.workflow_id, event.result))
    effects.extend(
        (
            Notify(state.workflow_id, event.result.notification),
            ReopenParent(state.workflow_id),
        )
    )
    return _finish(state, *effects)


def _workflow_failed(
    state: HomeEditState,
    raw_event: HomeEditEvent,
) -> HomeEditTransition:
    """Return the workflow failed value owned by this home edit workflow module."""
    event = raw_event
    assert isinstance(event, WorkflowFailed)
    return _finish(
        state,
        Notify(state.workflow_id, event.notification),
        ReopenParent(state.workflow_id),
    )


_Reducer = Callable[[HomeEditState, HomeEditEvent], HomeEditTransition]
_REDUCERS: dict[type[HomeEditEvent], _Reducer] = {
    TargetResolved: _target_resolved,
    SelectorResolved: _selector_resolved,
    TargetSelected: _target_selected,
    TargetCommitFinished: _target_commit_finished,
    ModelSelected: _model_selected,
    ManifestStageFinished: _manifest_stage_finished,
    SessionSelected: _session_selected,
    CheckpointSelected: _checkpoint_selected,
    DetailResolved: _detail_resolved,
    DetailSubmitted: _detail_submitted,
    DeleteConfirmed: _delete_confirmed,
    DeleteFinished: _delete_finished,
    SaveFinished: _save_finished,
    WorkflowFailed: _workflow_failed,
}


def reduce_home_edit(
    state: HomeEditState,
    event: HomeEditEvent,
) -> HomeEditTransition:
    """Apply one event, ignoring stale, terminal, or unsupported input."""

    if event.workflow_id != state.workflow_id:
        return _unchanged(state)
    if state.phase is HomeEditPhase.FINISHED:
        return _unchanged(state)
    reducer = _REDUCERS.get(type(event))
    if reducer is None:
        return _unchanged(state)
    return reducer(state, event)
