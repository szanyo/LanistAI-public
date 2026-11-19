# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate run precheck."""

from collections.abc import Callable
from dataclasses import dataclass

from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.checklist_service import (
    LoadChecklistResult,
    RuntimeChecklistResult,
)


@dataclass
class RunStartPrecheck:
    """Carry the data required for run start precheck."""

    can_start: bool
    message: str | None = None
    title: str = "Action blocked"
    severity: str = "warning"
    require_dialog: bool = False
    require_load_model: bool = False
    require_apply_configuration: bool = False


@dataclass
class RunStartExecutionPlan:
    """Describe the executable plan for run start execution."""

    action: str  # "blocked" | "resume" | "start"
    message: str | None = None
    title: str = "Action blocked"
    severity: str = "warning"
    require_dialog: bool = False
    require_load_model: bool = False
    require_apply_configuration: bool = False


@dataclass
class ActionPrecheck:
    """Carry the data required for action precheck."""

    allowed: bool
    message: str | None = None
    title: str = "Action blocked"
    severity: str = "warning"


class RunPrecheckService:
    """Coordinate run precheck operations."""

    def __init__(
        self,
        *,
        selected_manifest_provider: Callable[[], ModelManifest | None],
        active_manifest_provider: Callable[[], ModelManifest | None],
        has_current_model_provider: Callable[[], bool],
        active_configuration_provider: Callable[[], object],
        current_framework_provider: Callable[[], str],
        framework_normalizer: Callable[[str | None], str],
        model_load_checklist_builder: Callable[[object], LoadChecklistResult],
        framework_runtime_checklist_builder: Callable[[object, str], RuntimeChecklistResult],
        current_model_structure_validator: Callable[[], bool],
        session_closed_for_training_checker: Callable[[str | None], bool],
        session_checkpoint_checker: Callable[[str], bool],
        session_training_source_checker: Callable[[str | None], bool],
        session_framework_uuid_provider: Callable[[str], str],
        checkpoint_selection_checker: Callable[[str, str, str], bool] = (
            lambda _session, _checkpoint, _framework: False
        ),
        administration_block_reason_provider: Callable[[], str | None] = lambda: None,
        native_gui_mutation_guard: Callable[[], None] = lambda: None,
    ) -> None:
        """Initialize the run precheck service and its owned state."""
        self._selected_manifest_provider = selected_manifest_provider
        self._active_manifest_provider = active_manifest_provider
        self._has_current_model_provider = has_current_model_provider
        self._active_configuration_provider = active_configuration_provider
        self._current_framework_provider = current_framework_provider
        self._framework_normalizer = framework_normalizer
        self._model_load_checklist_builder = model_load_checklist_builder
        self._framework_runtime_checklist_builder = framework_runtime_checklist_builder
        self._current_model_structure_validator = current_model_structure_validator
        self._session_closed_for_training_checker = session_closed_for_training_checker
        self._session_checkpoint_checker = session_checkpoint_checker
        self._session_training_source_checker = session_training_source_checker
        self._session_framework_uuid_provider = session_framework_uuid_provider
        self._checkpoint_selection_checker = checkpoint_selection_checker
        self._administration_block_reason_provider = administration_block_reason_provider
        self._native_gui_mutation_guard = native_gui_mutation_guard

    def _has_training_source(
        self,
        selected_session: str | None,
        selected_checkpoint: str | None,
    ) -> bool:
        """Return whether training source applies to this run precheck service."""
        return bool(selected_checkpoint) or self._session_training_source_checker(selected_session)

    def evaluate_run_start(
        self,
        *,
        mode: RLModelRunningModes,
        selected_session: str | None,
        selected_checkpoint: str | None,
        has_previous_models: bool,
    ) -> RunStartPrecheck:
        """Evaluate the run start owned by this run precheck service."""
        administration_block_reason = self._administration_block_reason_provider()
        if administration_block_reason:
            return RunStartPrecheck(
                can_start=False,
                message=administration_block_reason,
                title="Recovery required",
                severity="error",
            )

        selected_manifest = self._selected_manifest_provider()
        precheck = self._selected_training_session_precheck(
            mode=mode,
            selected_manifest=selected_manifest,
            selected_session=selected_session,
        )
        if precheck is not None:
            return precheck
        precheck = self._active_model_precheck(
            has_current_model=self._has_current_model_provider(),
            has_previous_models=has_previous_models,
            selected_manifest=selected_manifest,
        )
        if precheck is not None:
            return precheck
        precheck = self._run_mode_precheck(
            mode=mode,
            selected_session=selected_session,
        )
        if precheck is not None:
            return precheck
        precheck = self._active_runtime_precheck()
        if precheck is not None:
            return precheck
        precheck = self._policy_checkpoint_start_precheck(
            mode=mode,
            selected_session=selected_session,
            selected_checkpoint=selected_checkpoint,
        )
        if precheck is not None:
            return precheck
        precheck = self._training_start_precheck(
            mode=mode,
            selected_session=selected_session,
            selected_checkpoint=selected_checkpoint,
        )
        if precheck is not None:
            return precheck

        return RunStartPrecheck(can_start=True)

    @staticmethod
    def _selected_training_session_precheck(
        *,
        mode: RLModelRunningModes,
        selected_manifest: ModelManifest | None,
        selected_session: str | None,
    ) -> RunStartPrecheck | None:
        """Return the selected training session precheck value owned by this run precheck service."""
        if mode != RLModelRunningModes.TRAINING or selected_manifest is None or selected_session:
            return None
        return RunStartPrecheck(
            can_start=False,
            message=(
                "A model is selected, but no training session is selected yet. "
                "Create/fork a session in Configure, apply with Confirm, then start."
            ),
            title="Training blocked",
            severity="warning",
            require_dialog=True,
        )

    @staticmethod
    def _active_model_precheck(
        *,
        has_current_model: bool,
        has_previous_models: bool,
        selected_manifest: ModelManifest | None,
    ) -> RunStartPrecheck | None:
        """Return the active model precheck value owned by this run precheck service."""
        if has_current_model:
            return None
        if has_previous_models and selected_manifest is not None:
            return RunStartPrecheck(
                can_start=False,
                message=(
                    "A model is selected, but it is not active yet. "
                    "Applying configuration before starting."
                ),
                title="Configuration required",
                severity="information",
                require_apply_configuration=True,
            )
        if has_previous_models:
            return RunStartPrecheck(
                can_start=False,
                message="No active model loaded. Select model from dialog, then start.",
                title="Load required",
                severity="information",
                require_load_model=True,
            )
        return RunStartPrecheck(
            can_start=False,
            message="Progress cannot start: No model selected.",
            title="Error",
            severity="error",
            require_dialog=True,
        )

    @staticmethod
    def _run_mode_precheck(
        *,
        mode: RLModelRunningModes,
        selected_session: str | None,
    ) -> RunStartPrecheck | None:
        """Return the run mode precheck value owned by this run precheck service."""
        if mode == RLModelRunningModes.EVALUATION and not selected_session:
            return RunStartPrecheck(
                can_start=False,
                message=(
                    "Evaluation requires a selected model session and checkpoint. "
                    "Choose them in Configure, then apply with Confirm."
                ),
                title="Evaluation blocked",
                severity="warning",
                require_dialog=True,
            )
        if mode != RLModelRunningModes.INFERENCE or selected_session:
            return None
        return RunStartPrecheck(
            can_start=False,
            message=(
                "Inference requires a selected model session and checkpoint. "
                "Choose them in Configure, then apply with Confirm."
            ),
            title="Inference blocked",
            severity="warning",
            require_dialog=True,
        )

    def _policy_checkpoint_start_precheck(
        self,
        *,
        mode: RLModelRunningModes,
        selected_session: str | None,
        selected_checkpoint: str | None,
    ) -> RunStartPrecheck | None:
        """Revalidate the selected policy checkpoint before inference or evaluation."""
        if mode not in (RLModelRunningModes.EVALUATION, RLModelRunningModes.INFERENCE):
            return None
        operation = "Evaluation" if mode == RLModelRunningModes.EVALUATION else "Inference"
        if not selected_session or not selected_checkpoint:
            return RunStartPrecheck(
                can_start=False,
                message=(
                    f"{operation} requires an explicitly selected checkpoint. "
                    "Choose one in Configure, then apply with Confirm."
                ),
                title=f"{operation} blocked",
                severity="warning",
                require_dialog=True,
            )
        framework_uuid = self._framework_normalizer(self._current_framework_provider())
        if self._checkpoint_selection_checker(
            selected_session,
            selected_checkpoint,
            framework_uuid,
        ):
            return None
        return RunStartPrecheck(
            can_start=False,
            message=(
                f"The selected {operation.lower()} checkpoint is no longer valid for "
                "the selected session and framework. Choose it again in Configure."
            ),
            title=f"{operation} blocked",
            severity="warning",
            require_dialog=True,
        )

    def _active_runtime_precheck(self) -> RunStartPrecheck | None:
        """Return the active runtime precheck value owned by this run precheck service."""
        active_manifest = self._active_manifest_provider()
        if active_manifest is None:
            return None
        structure_checklist = self._model_load_checklist_builder(active_manifest)
        if not structure_checklist.valid:
            return RunStartPrecheck(
                can_start=False,
                message=structure_checklist.summary,
                title="Model structure check failed",
                severity="error",
                require_dialog=True,
            )
        framework_uuid = self._resolve_active_framework(active_manifest)
        if not framework_uuid:
            return RunStartPrecheck(
                can_start=False,
                message=("Model runtime configuration is incomplete: missing active framework."),
                title="Runtime check failed",
                severity="error",
                require_dialog=True,
            )
        runtime_checklist = self._framework_runtime_checklist_builder(
            active_manifest,
            framework_uuid,
        )
        if runtime_checklist.valid:
            return None
        return RunStartPrecheck(
            can_start=False,
            message=runtime_checklist.summary,
            title=f"{framework_uuid.upper()} runtime check failed",
            severity="error",
            require_dialog=True,
        )

    def _resolve_active_framework(self, active_manifest: ModelManifest) -> str:
        """Resolve the active framework required by this run precheck service."""
        framework_uuid = self._current_framework_provider()
        if framework_uuid:
            return framework_uuid
        configuration = active_manifest.get_configuration()
        return self._framework_normalizer(
            getattr(configuration, "active_runtime_framework_uuid", None)
        ) or self._framework_normalizer(getattr(configuration, "preferred_framework_uuid", None))

    def _training_start_precheck(
        self,
        *,
        mode: RLModelRunningModes,
        selected_session: str | None,
        selected_checkpoint: str | None,
    ) -> RunStartPrecheck | None:
        """Return the training start precheck value owned by this run precheck service."""
        if mode != RLModelRunningModes.TRAINING:
            return None
        if not selected_session:
            return RunStartPrecheck(
                can_start=False,
                message=(
                    "Training requires a selected session. Use Configure to choose the next step."
                ),
                title="Training blocked",
                severity="warning",
                require_dialog=True,
            )
        has_training_source = self._has_training_source(
            selected_session,
            selected_checkpoint,
        )
        if not self._session_closed_for_training_checker(selected_session) or has_training_source:
            return None
        return RunStartPrecheck(
            can_start=False,
            message=(
                "The selected session is already closed for training. "
                "Create a new session before starting training again."
            ),
            title="Training blocked",
            severity="warning",
            require_dialog=True,
        )

    def build_run_start_execution_plan(
        self,
        *,
        mode: RLModelRunningModes,
        selected_session: str | None,
        selected_checkpoint: str | None,
        has_previous_models: bool,
        panel_state: RLModelState,
    ) -> RunStartExecutionPlan:
        """Build the run start execution plan required by this run precheck service."""
        self._native_gui_mutation_guard()
        precheck = self.evaluate_run_start(
            mode=mode,
            selected_session=selected_session,
            selected_checkpoint=selected_checkpoint,
            has_previous_models=has_previous_models,
        )

        if not precheck.can_start:
            return RunStartExecutionPlan(
                action="blocked",
                message=precheck.message,
                title=precheck.title,
                severity=precheck.severity,
                require_dialog=precheck.require_dialog,
                require_load_model=precheck.require_load_model,
                require_apply_configuration=precheck.require_apply_configuration,
            )

        if panel_state == RLModelState.PAUSED:
            return RunStartExecutionPlan(action="resume")

        return RunStartExecutionPlan(action="start")

    @staticmethod
    def evaluate_create_model(mode: RLModelRunningModes) -> ActionPrecheck:
        """Evaluate the create model owned by this run precheck service."""
        if mode != RLModelRunningModes.TRAINING:
            return ActionPrecheck(
                allowed=False,
                message="Create model is only available in Training phase.",
                title="Action blocked",
                severity="warning",
            )

        return ActionPrecheck(allowed=True)

    def evaluate_create_session(self, mode: RLModelRunningModes) -> ActionPrecheck:
        """Evaluate the create session owned by this run precheck service."""
        if mode != RLModelRunningModes.TRAINING:
            return ActionPrecheck(
                allowed=False,
                message="Create session is only available in Training phase.",
                title="Action blocked",
                severity="warning",
            )

        if self._active_manifest_provider() is None:
            return ActionPrecheck(
                allowed=False,
                message="Select a model before creating a new session.",
                title="Action blocked",
                severity="warning",
            )

        if not self._current_model_structure_validator():
            return ActionPrecheck(
                allowed=False,
                message="A structurally valid model is required before creating a new session.",
                title="Action blocked",
                severity="warning",
            )

        return ActionPrecheck(allowed=True)

    def evaluate_configuration_confirm(
        self,
        *,
        selected_mode: RLModelRunningModes,
        selected_framework_uuid: str,
        selected_session_name: str | None,
        selected_checkpoint_name: str | None,
    ) -> ActionPrecheck:
        """Evaluate the configuration confirm owned by this run precheck service."""
        manifest = self._selected_manifest_provider()
        model_precheck = self._configuration_model_precheck(
            manifest=manifest,
            selected_framework_uuid=selected_framework_uuid,
        )
        if model_precheck is not None:
            return model_precheck
        selection_precheck = self._configuration_selection_precheck(
            selected_mode=selected_mode,
            selected_framework_uuid=selected_framework_uuid,
            selected_session_name=selected_session_name,
        )
        if selection_precheck is not None:
            return selection_precheck
        assert selected_session_name is not None
        policy_precheck = self._policy_configuration_precheck(
            selected_mode=selected_mode,
            selected_framework_uuid=selected_framework_uuid,
            selected_session_name=selected_session_name,
            selected_checkpoint_name=selected_checkpoint_name,
        )
        if policy_precheck is not None:
            return policy_precheck
        training_precheck = self._training_configuration_precheck(
            selected_mode=selected_mode,
            selected_session_name=selected_session_name,
            selected_checkpoint_name=selected_checkpoint_name,
        )
        if training_precheck is not None:
            return training_precheck

        return ActionPrecheck(allowed=True)

    def _configuration_model_precheck(
        self,
        *,
        manifest: ModelManifest | None,
        selected_framework_uuid: str,
    ) -> ActionPrecheck | None:
        """Return the configuration model precheck value owned by this run precheck service."""
        if manifest is None:
            return ActionPrecheck(
                allowed=False,
                message="Select a model before applying configuration.",
                title="Configuration blocked",
                severity="warning",
            )
        structure_checklist = self._model_load_checklist_builder(manifest)
        if not structure_checklist.valid:
            return ActionPrecheck(
                allowed=False,
                message="The selected model is structurally incomplete.",
                title="Configuration blocked",
                severity="warning",
            )
        runtime_checklist = self._framework_runtime_checklist_builder(
            manifest,
            selected_framework_uuid,
        )
        if runtime_checklist.valid:
            return None
        return ActionPrecheck(
            allowed=False,
            message=runtime_checklist.summary,
            title="Configuration blocked",
            severity="warning",
        )

    def _configuration_selection_precheck(
        self,
        *,
        selected_mode: RLModelRunningModes,
        selected_framework_uuid: str,
        selected_session_name: str | None,
    ) -> ActionPrecheck | None:
        """Return the configuration selection precheck value owned by this run precheck service."""
        if not selected_session_name:
            return ActionPrecheck(
                allowed=False,
                message="Select a valid session before applying configuration.",
                title="Configuration blocked",
                severity="warning",
            )
        normalized_framework = self._framework_normalizer(selected_framework_uuid)
        session_framework = self._framework_normalizer(
            self._session_framework_uuid_provider(selected_session_name)
        )
        if not session_framework or session_framework == normalized_framework:
            return None
        return ActionPrecheck(
            allowed=False,
            message=(
                f"The selected session belongs to '{session_framework}', "
                f"but the selected framework is '{normalized_framework}'. "
                "Select a session created for the chosen framework."
            ),
            title="Configuration blocked",
            severity="warning",
        )

    def _policy_configuration_precheck(
        self,
        *,
        selected_mode: RLModelRunningModes,
        selected_framework_uuid: str,
        selected_session_name: str,
        selected_checkpoint_name: str | None,
    ) -> ActionPrecheck | None:
        """Require a valid policy checkpoint for inference and evaluation."""
        if selected_mode not in (
            RLModelRunningModes.EVALUATION,
            RLModelRunningModes.INFERENCE,
        ):
            return None
        operation = "Evaluation" if selected_mode == RLModelRunningModes.EVALUATION else "Inference"
        if not selected_checkpoint_name:
            return ActionPrecheck(
                allowed=False,
                message=f"Select a checkpoint before applying {operation} configuration.",
                title=f"{operation} blocked",
                severity="warning",
            )
        normalized_framework = self._framework_normalizer(selected_framework_uuid)
        if self._checkpoint_selection_checker(
            selected_session_name,
            selected_checkpoint_name,
            normalized_framework,
        ):
            return None
        return ActionPrecheck(
            allowed=False,
            message=(
                "The selected checkpoint is not valid for this session and "
                "framework. Select an existing checkpoint."
            ),
            title=f"{operation} blocked",
            severity="warning",
        )

    def _training_configuration_precheck(
        self,
        *,
        selected_mode: RLModelRunningModes,
        selected_session_name: str,
        selected_checkpoint_name: str | None,
    ) -> ActionPrecheck | None:
        """Return the training configuration precheck value owned by this run precheck service."""
        if selected_mode != RLModelRunningModes.TRAINING:
            return None
        has_training_source = self._has_training_source(
            selected_session_name,
            selected_checkpoint_name,
        )
        if self._session_checkpoint_checker(selected_session_name) and not has_training_source:
            return ActionPrecheck(
                allowed=False,
                message=(
                    "The selected session already contains checkpoints. Select a "
                    "checkpoint to continue training or create a new forked session."
                ),
                title="Configuration blocked",
                severity="warning",
            )
        if (
            not self._session_closed_for_training_checker(selected_session_name)
            or has_training_source
        ):
            return None
        return ActionPrecheck(
            allowed=False,
            message=(
                "The selected session is already closed for training. "
                "Create a new session before starting training again."
            ),
            title="Configuration blocked",
            severity="warning",
        )
