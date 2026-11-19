# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate configuration runtime state."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.rl_model.metadata.configuration_schema import ConfigurationSchema
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.checklist_service import (
    ChecklistItem,
    LoadChecklistResult,
    RuntimeChecklistResult,
)
from lanista.reinforcengine.rl_model.runtime.run_precheck_service import (
    ActionPrecheck,
    RunStartPrecheck,
)


class RuntimeSelectionView(Protocol):
    """Selection projection required to assemble immutable runtime state."""

    @property
    def session_name(self) -> str | None:
        """Return the session name exposed by this runtime selection view."""
        ...

    @property
    def checkpoint_name(self) -> str | None:
        """Return the checkpoint name exposed by this runtime selection view."""
        ...


@dataclass(frozen=True, slots=True)
class ConfigurationSelectionPreference:
    """Presence-aware session/checkpoint choice supplied by a UI workflow."""

    session_name: str | None
    checkpoint_name: str | None


@dataclass(frozen=True, slots=True)
class ConfigurationStatusItem:
    """Carry the data required for configuration status item."""

    label: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ConfigurationFeedbackSignal:
    """Carry the data required for configuration feedback signal."""

    key: str
    label: str
    ok: bool
    detail: str = ""
    blocked_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigurationFrameworkStatus:
    """Carry the data required for configuration framework status."""

    framework_uuid: str
    available: bool
    detail: str
    checklist_items: list[ChecklistItem]


@dataclass(frozen=True, slots=True)
class ConfigurationRuntimeState:
    """Represent runtime state for configuration runtime."""

    has_valid_model: bool
    loaded_model_name: str | None
    loaded_model_uuid: str | None
    can_evaluate: bool
    can_new_session: bool
    can_confirm: bool
    can_start_run: bool
    selected_mode: RLModelRunningModes
    selected_framework_uuid: str
    session_options: list[tuple[str, str]]
    selected_session_name: str | None
    checkpoint_options: list[tuple[str, str]]
    selected_checkpoint_name: str | None
    configuration_status_items: list[ConfigurationStatusItem]
    framework_statuses: list[ConfigurationFrameworkStatus]
    structure_check_items: list[ChecklistItem]
    runtime_check_items: list[ChecklistItem]
    configuration_locked: bool = False


@dataclass(frozen=True, slots=True)
class _RuntimeStateSources:
    """Single-read source snapshot used by one state assembly."""

    runtime_selection: RuntimeSelectionView
    active_manifest: object | None
    active_configuration: object | None
    loaded_model_name: str | None
    loaded_model_uuid: str | None
    structure_checklist: LoadChecklistResult | None
    has_loaded_model: bool
    has_valid_model: bool
    configuration_locked: bool


@dataclass(frozen=True, slots=True)
class _RuntimeBranchResolution:
    """Framework, mode, and runtime-checklist choices for one build."""

    selected_mode: RLModelRunningModes
    selected_framework_uuid: str
    can_evaluate: bool
    session_options: list[tuple[str, str]]
    runtime_checklist: RuntimeChecklistResult | None


@dataclass(frozen=True, slots=True)
class _SelectionResolution:
    """Resolved session and checkpoint values plus their option lists."""

    selected_session_name: str | None
    checkpoint_options: list[tuple[str, str]]
    selected_checkpoint_name: str | None


@dataclass(frozen=True, slots=True)
class _ActionEvaluations:
    """Precheck results evaluated against the same immutable choices."""

    confirm: ActionPrecheck
    create_model: ActionPrecheck
    create_session: ActionPrecheck
    run_start: RunStartPrecheck


class ConfigurationRuntimeStateService:
    """Coordinate configuration runtime state operations."""

    def __init__(
        self,
        *,
        runtime_selection_provider: Callable[[], RuntimeSelectionView],
        active_manifest_provider: Callable[[], object | None],
        active_configuration_provider: Callable[[], object | None],
        current_model_display_name_provider: Callable[[], str | None],
        current_framework_provider: Callable[[], str],
        model_load_checklist_builder: Callable[[object], LoadChecklistResult],
        framework_runtime_checklist_builder: Callable[[object, str], RuntimeChecklistResult],
        session_options_provider: Callable[[str | None], list[tuple[str, str]]],
        checkpoint_options_provider: Callable[[str | None, str | None], list[tuple[str, str]]],
        can_evaluate_provider: Callable[[str | None], bool],
        confirm_precheck_evaluator: Callable[
            [RLModelRunningModes, str, str | None, str | None], ActionPrecheck
        ],
        create_model_precheck_evaluator: Callable[[RLModelRunningModes], ActionPrecheck],
        create_session_precheck_evaluator: Callable[[RLModelRunningModes], ActionPrecheck],
        run_start_precheck_evaluator: Callable[
            [RLModelRunningModes, str | None, str | None, bool], RunStartPrecheck
        ],
        configuration_locked_provider: Callable[[], bool] = lambda: False,
    ) -> None:
        """Initialize the configuration runtime state service and its owned state."""
        self._runtime_selection_provider = runtime_selection_provider
        self._active_manifest_provider = active_manifest_provider
        self._active_configuration_provider = active_configuration_provider
        self._current_model_display_name_provider = current_model_display_name_provider
        self._current_framework_provider = current_framework_provider
        self._model_load_checklist_builder = model_load_checklist_builder
        self._framework_runtime_checklist_builder = framework_runtime_checklist_builder
        self._session_options_provider = session_options_provider
        self._checkpoint_options_provider = checkpoint_options_provider
        self._can_evaluate_provider = can_evaluate_provider
        self._confirm_precheck_evaluator = confirm_precheck_evaluator
        self._create_model_precheck_evaluator = create_model_precheck_evaluator
        self._create_session_precheck_evaluator = create_session_precheck_evaluator
        self._run_start_precheck_evaluator = run_start_precheck_evaluator
        self._configuration_locked_provider = configuration_locked_provider

    @staticmethod
    def _build_action_feedback_signal(
        key: str,
        label: str,
        precheck: ActionPrecheck,
        ready_detail: str = "ready",
    ) -> ConfigurationFeedbackSignal:
        """Build the action feedback signal required by this configuration runtime state service."""
        reason = str(precheck.message or "blocked").strip()
        if precheck.allowed:
            return ConfigurationFeedbackSignal(
                key=key,
                label=label,
                ok=True,
                detail=ready_detail,
                blocked_reason=None,
            )
        return ConfigurationFeedbackSignal(
            key=key,
            label=label,
            ok=False,
            detail="blocked",
            blocked_reason=reason,
        )

    @staticmethod
    def _feedback_signal_to_status_item(
        signal: ConfigurationFeedbackSignal,
    ) -> ConfigurationStatusItem:
        """Return the feedback signal to status item value owned by this configuration runtime
        state service.
        """
        detail = signal.detail
        if not signal.ok and signal.blocked_reason:
            detail = signal.blocked_reason
        return ConfigurationStatusItem(
            label=signal.label,
            ok=signal.ok,
            detail=detail,
        )

    @staticmethod
    def _framework_paths(configuration: object) -> dict[str, str]:
        """Return the framework paths value owned by this configuration runtime state service."""
        framework_resource_paths = (
            getattr(configuration, "framework_resource_paths", None)
            if configuration is not None
            else None
        )
        if isinstance(framework_resource_paths, dict):
            return {
                framework_uuid: str(framework_resource_paths.get(framework_uuid, "") or "")
                for framework_uuid in ("ray", "sb3")
            }
        return {
            framework_uuid: str(getattr(framework_resource_paths, framework_uuid, "") or "")
            for framework_uuid in ("ray", "sb3")
        }

    @staticmethod
    def _configured_framework_uuid(configuration: object) -> str:
        """Return the configured framework uuid value owned by this configuration runtime state
        service.
        """
        schema_source = getattr(configuration, "model_configuration", configuration)
        schema = ConfigurationSchema.from_configuration(schema_source)
        preferred_framework_uuid = (
            str(
                getattr(configuration, "preferred_framework_uuid", "")
                or schema.preferred_framework_uuid
                or ""
            )
            .strip()
            .lower()
        )
        if preferred_framework_uuid in {"ray", "sb3"}:
            return preferred_framework_uuid
        configured_framework_uuid = (
            str(
                getattr(configuration, "active_runtime_framework_uuid", "")
                or schema.active_runtime_framework_uuid
                or ""
            )
            .strip()
            .lower()
        )
        return configured_framework_uuid if configured_framework_uuid in {"ray", "sb3"} else "ray"

    @staticmethod
    def _normalize_selection_preference(
        preference: ConfigurationSelectionPreference | None,
    ) -> ConfigurationSelectionPreference | None:
        """Normalize the selection preference accepted by this configuration runtime state service.
        """
        if preference is None:
            return None
        return ConfigurationSelectionPreference(
            session_name=str(preference.session_name or "").strip() or None,
            checkpoint_name=str(preference.checkpoint_name or "").strip() or None,
        )

    def _capture_sources(self) -> _RuntimeStateSources:
        """Capture the sources required by this configuration runtime state service."""
        runtime_selection = self._runtime_selection_provider()
        active_manifest = self._active_manifest_provider()
        has_loaded_model = active_manifest is not None
        loaded_model_name = self._current_model_display_name_provider()
        active_configuration = self._active_configuration_provider()
        loaded_model_uuid = (
            str(getattr(active_configuration, "model_uuid", "") or "").strip() or None
            if has_loaded_model
            else None
        )
        structure_checklist = (
            self._model_load_checklist_builder(active_manifest)
            if active_manifest is not None
            else None
        )
        return _RuntimeStateSources(
            runtime_selection=runtime_selection,
            active_manifest=active_manifest,
            active_configuration=active_configuration,
            loaded_model_name=loaded_model_name,
            loaded_model_uuid=loaded_model_uuid,
            structure_checklist=structure_checklist,
            has_loaded_model=has_loaded_model,
            has_valid_model=bool(structure_checklist and structure_checklist.valid),
            configuration_locked=bool(self._configuration_locked_provider()),
        )

    def _resolve_runtime_branch(
        self,
        *,
        sources: _RuntimeStateSources,
        preferred_mode: RLModelRunningModes,
        preferred_framework_uuid: str,
        framework_options: list[tuple[str, str]],
    ) -> _RuntimeBranchResolution:
        """Resolve the runtime branch required by this configuration runtime state service."""
        valid_framework_values = {value for _, value in framework_options}
        configured_framework_uuid = (
            self._configured_framework_uuid(sources.active_configuration)
            if sources.active_configuration is not None
            else "ray"
        )
        selected_framework_uuid = (
            preferred_framework_uuid
            if preferred_framework_uuid in valid_framework_values
            else configured_framework_uuid
        )
        if selected_framework_uuid not in valid_framework_values:
            selected_framework_uuid = framework_options[0][1] if framework_options else "ray"
        runtime_checklist = (
            self._framework_runtime_checklist_builder(
                sources.active_manifest,
                selected_framework_uuid,
            )
            if sources.active_manifest is not None
            else None
        )
        session_options = (
            self._session_options_provider(selected_framework_uuid)
            if sources.has_loaded_model
            else []
        )
        can_evaluate = (
            self._can_evaluate_provider(selected_framework_uuid)
            if sources.has_loaded_model
            else False
        )
        selected_mode = preferred_mode
        if selected_mode == RLModelRunningModes.EVALUATION and not can_evaluate:
            selected_mode = RLModelRunningModes.TRAINING
        return _RuntimeBranchResolution(
            selected_mode=selected_mode,
            selected_framework_uuid=selected_framework_uuid,
            can_evaluate=can_evaluate,
            session_options=session_options,
            runtime_checklist=runtime_checklist,
        )

    def _resolve_selection(
        self,
        *,
        sources: _RuntimeStateSources,
        branch: _RuntimeBranchResolution,
        preference: ConfigurationSelectionPreference | None,
    ) -> _SelectionResolution:
        """Resolve the selection required by this configuration runtime state service."""
        valid_sessions = {value for _, value in branch.session_options}
        selected_session = (
            preference.session_name
            if preference is not None
            else sources.runtime_selection.session_name
        )
        if selected_session and selected_session not in valid_sessions:
            selected_session = None
        if selected_session is None and branch.session_options and preference is None:
            selected_session = branch.session_options[0][1]
        checkpoint_options = (
            self._checkpoint_options_provider(
                selected_session,
                branch.selected_framework_uuid,
            )
            if sources.has_loaded_model
            else []
        )
        valid_checkpoints = {value for _, value in checkpoint_options}
        selected_checkpoint = (
            preference.checkpoint_name
            if preference is not None
            else sources.runtime_selection.checkpoint_name
        )
        if selected_checkpoint and selected_checkpoint not in valid_checkpoints:
            selected_checkpoint = None
        return _SelectionResolution(
            selected_session_name=selected_session,
            checkpoint_options=checkpoint_options,
            selected_checkpoint_name=selected_checkpoint,
        )

    def _evaluate_actions(
        self,
        *,
        sources: _RuntimeStateSources,
        branch: _RuntimeBranchResolution,
        selection: _SelectionResolution,
    ) -> _ActionEvaluations:
        """Evaluate the actions owned by this configuration runtime state service."""
        mode = branch.selected_mode
        return _ActionEvaluations(
            confirm=self._confirm_precheck_evaluator(
                mode,
                branch.selected_framework_uuid,
                selection.selected_session_name,
                selection.selected_checkpoint_name,
            ),
            create_model=self._create_model_precheck_evaluator(mode),
            create_session=self._create_session_precheck_evaluator(mode),
            run_start=self._run_start_precheck_evaluator(
                mode,
                selection.selected_session_name,
                selection.selected_checkpoint_name,
                sources.has_loaded_model,
            ),
        )

    def _build_framework_statuses(
        self,
        sources: _RuntimeStateSources,
    ) -> list[ConfigurationFrameworkStatus]:
        """Build the framework statuses required by this configuration runtime state service."""
        framework_paths = self._framework_paths(sources.active_configuration)
        statuses: list[ConfigurationFrameworkStatus] = []
        for framework_uuid in ("ray", "sb3"):
            checklist = (
                self._framework_runtime_checklist_builder(
                    sources.active_manifest,
                    framework_uuid,
                )
                if sources.active_manifest is not None
                else None
            )
            checklist_items: list[ChecklistItem] = checklist.items if checklist else []
            detail = (
                f"{sum(1 for item in checklist_items if item.exists)}/{len(checklist_items)}"
                if checklist is not None
                else "-"
            )
            statuses.append(
                ConfigurationFrameworkStatus(
                    framework_uuid=framework_uuid,
                    available=(
                        os.path.isdir(str(framework_paths.get(framework_uuid, "") or ""))
                        if sources.has_loaded_model
                        else False
                    ),
                    detail=detail,
                    checklist_items=checklist_items,
                )
            )
        return statuses

    @staticmethod
    def _checklist_progress(items: list[ChecklistItem]) -> str:
        """Return the checklist progress value owned by this configuration runtime state service."""
        if not items:
            return "-"
        return f"{sum(1 for item in items if item.exists)}/{len(items)}"

    def _build_model_feedback(
        self,
        *,
        sources: _RuntimeStateSources,
        branch: _RuntimeBranchResolution,
        detected_framework_uuid: str | None,
    ) -> list[ConfigurationFeedbackSignal]:
        """Build the model feedback required by this configuration runtime state service."""
        structure_items = sources.structure_checklist.items if sources.structure_checklist else []
        return [
            ConfigurationFeedbackSignal(
                key="model-detected",
                label="Model detected",
                ok=sources.has_loaded_model,
                detail=sources.loaded_model_name or "-",
            ),
            ConfigurationFeedbackSignal(
                key="model-uuid",
                label="Model UUID",
                ok=bool(sources.loaded_model_uuid),
                detail=sources.loaded_model_uuid or "-",
            ),
            ConfigurationFeedbackSignal(
                key="structure-check",
                label="Structure check",
                ok=sources.has_valid_model,
                detail=self._checklist_progress(structure_items),
            ),
            ConfigurationFeedbackSignal(
                key="active-framework",
                label="Active framework branch",
                ok=bool(detected_framework_uuid),
                detail=detected_framework_uuid or branch.selected_framework_uuid or "-",
            ),
        ]

    def _build_selection_feedback(
        self,
        *,
        branch: _RuntimeBranchResolution,
        selection: _SelectionResolution,
    ) -> list[ConfigurationFeedbackSignal]:
        """Build the selection feedback required by this configuration runtime state service."""
        session_count = len(branch.session_options)
        checkpoint_count = len(selection.checkpoint_options)
        session_selected = bool(selection.selected_session_name)
        runtime_items = branch.runtime_checklist.items if branch.runtime_checklist else []
        return [
            ConfigurationFeedbackSignal(
                "sessions-detected", "Sessions detected", session_count > 0, str(session_count)
            ),
            ConfigurationFeedbackSignal(
                "selected-session",
                "Selected session",
                session_selected,
                selection.selected_session_name or "-",
            ),
            ConfigurationFeedbackSignal(
                "checkpoints-detected",
                "Checkpoints detected",
                checkpoint_count > 0,
                str(checkpoint_count) if session_selected else "-",
            ),
            ConfigurationFeedbackSignal(
                "session-checkpoints",
                "Session checkpoints",
                session_selected,
                str(checkpoint_count) if session_selected else "-",
            ),
            ConfigurationFeedbackSignal(
                "selected-checkpoint",
                "Selected checkpoint",
                bool(selection.selected_checkpoint_name),
                selection.selected_checkpoint_name or "fresh training",
            ),
            ConfigurationFeedbackSignal(
                "runtime-config",
                "Selected runtime config",
                bool(branch.runtime_checklist and branch.runtime_checklist.valid),
                self._checklist_progress(runtime_items),
            ),
        ]

    def _build_action_feedback(
        self,
        *,
        branch: _RuntimeBranchResolution,
        evaluations: _ActionEvaluations,
    ) -> list[ConfigurationFeedbackSignal]:
        """Build the action feedback required by this configuration runtime state service."""
        run_start = evaluations.run_start
        return [
            self._build_action_feedback_signal(
                "create-model", "Create model available", evaluations.create_model
            ),
            self._build_action_feedback_signal(
                "new-session", "New session available", evaluations.create_session
            ),
            self._build_action_feedback_signal(
                "confirm",
                {
                    RLModelRunningModes.TRAINING: "Training available",
                    RLModelRunningModes.EVALUATION: "Apply available",
                    RLModelRunningModes.INFERENCE: "Inference available",
                }.get(branch.selected_mode, "Confirm available"),
                evaluations.confirm,
            ),
            ConfigurationFeedbackSignal(
                key="run-start",
                label="Run start precheck",
                ok=run_start.can_start or run_start.require_apply_configuration,
                detail=(
                    "ready"
                    if run_start.can_start
                    else "apply required"
                    if run_start.require_apply_configuration
                    else "blocked"
                ),
                blocked_reason=(
                    str(run_start.message or "blocked").strip() if not run_start.can_start else None
                ),
            ),
        ]

    def _build_feedback_signals(
        self,
        *,
        sources: _RuntimeStateSources,
        branch: _RuntimeBranchResolution,
        selection: _SelectionResolution,
        evaluations: _ActionEvaluations,
        detected_framework_uuid: str | None,
    ) -> list[ConfigurationFeedbackSignal]:
        """Build the feedback signals required by this configuration runtime state service."""
        return [
            *self._build_model_feedback(
                sources=sources,
                branch=branch,
                detected_framework_uuid=detected_framework_uuid,
            ),
            *self._build_selection_feedback(branch=branch, selection=selection),
            *self._build_action_feedback(branch=branch, evaluations=evaluations),
        ]

    def _assemble_state(
        self,
        *,
        sources: _RuntimeStateSources,
        branch: _RuntimeBranchResolution,
        selection: _SelectionResolution,
        evaluations: _ActionEvaluations,
        feedback_signals: list[ConfigurationFeedbackSignal],
        framework_statuses: list[ConfigurationFrameworkStatus],
    ) -> ConfigurationRuntimeState:
        """Return the assemble state value owned by this configuration runtime state service."""
        structure_items = sources.structure_checklist.items if sources.structure_checklist else []
        runtime_items = branch.runtime_checklist.items if branch.runtime_checklist else []
        return ConfigurationRuntimeState(
            has_valid_model=sources.has_valid_model,
            loaded_model_name=sources.loaded_model_name,
            loaded_model_uuid=sources.loaded_model_uuid,
            can_evaluate=branch.can_evaluate,
            can_new_session=sources.has_valid_model and not sources.configuration_locked,
            can_confirm=evaluations.confirm.allowed and not sources.configuration_locked,
            can_start_run=(
                not sources.configuration_locked
                and (
                    evaluations.run_start.can_start
                    or evaluations.run_start.require_apply_configuration
                )
            ),
            configuration_locked=sources.configuration_locked,
            selected_mode=branch.selected_mode,
            selected_framework_uuid=branch.selected_framework_uuid,
            session_options=branch.session_options,
            selected_session_name=selection.selected_session_name,
            checkpoint_options=selection.checkpoint_options,
            selected_checkpoint_name=selection.selected_checkpoint_name,
            configuration_status_items=[
                self._feedback_signal_to_status_item(signal) for signal in feedback_signals
            ],
            framework_statuses=framework_statuses,
            structure_check_items=structure_items,
            runtime_check_items=runtime_items,
        )

    def build(
        self,
        *,
        preferred_mode: RLModelRunningModes,
        preferred_framework_uuid: str,
        framework_options: list[tuple[str, str]],
        selection_preference: ConfigurationSelectionPreference | None = None,
    ) -> ConfigurationRuntimeState:
        """Build the configured result owned by this configuration runtime state service."""
        preference = self._normalize_selection_preference(selection_preference)
        sources = self._capture_sources()
        branch = self._resolve_runtime_branch(
            sources=sources,
            preferred_mode=preferred_mode,
            preferred_framework_uuid=preferred_framework_uuid,
            framework_options=framework_options,
        )
        selection = self._resolve_selection(
            sources=sources,
            branch=branch,
            preference=preference,
        )
        evaluations = self._evaluate_actions(
            sources=sources,
            branch=branch,
            selection=selection,
        )
        detected_framework_uuid = (
            self._current_framework_provider() if sources.has_loaded_model else None
        )
        framework_statuses = self._build_framework_statuses(sources)
        feedback_signals = self._build_feedback_signals(
            sources=sources,
            branch=branch,
            selection=selection,
            evaluations=evaluations,
            detected_framework_uuid=detected_framework_uuid,
        )
        return self._assemble_state(
            sources=sources,
            branch=branch,
            selection=selection,
            evaluations=evaluations,
            feedback_signals=feedback_signals,
            framework_statuses=framework_statuses,
        )
