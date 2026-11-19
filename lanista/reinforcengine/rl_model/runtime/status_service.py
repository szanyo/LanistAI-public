# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate runtime status."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from lanista.reinforcengine.rl_model.model import RLModelRunningModes


@dataclass
class RuntimeCapabilities:
    """Carry the data required for runtime capabilities."""

    has_training_session: bool
    has_evaluation_runtime: bool
    has_inference_runtime: bool = False


@dataclass
class ModelReadyPresentation:
    """Prepare presentation data for model ready."""

    activity: str
    step: str
    training_only_warning: bool


@dataclass
class RunModeExecutionPlan:
    """Describe the executable plan for run mode execution."""

    mode: RLModelRunningModes
    refresh_tensorboard_writer: bool
    start_tensorboard: bool


class RuntimeStatusService:
    """Coordinate runtime status operations."""

    def __init__(
        self,
        *,
        active_manifest_provider: Callable[[], object | None],
        loaded_manifest_provider: Callable[[], object | None],
        model_structure_validator: Callable[[object], bool],
        current_framework_provider: Callable[[], str | None],
        loaded_framework_provider: Callable[[], str | None] | None = None,
        sessions_provider: Callable[[str | None], list[str]],
        checkpoint_names_provider: Callable[[str, str | None], list[str]],
        last_model_load_issue_provider: Callable[[], str | None],
    ) -> None:
        """Initialize the runtime status service and its owned state."""
        self._active_manifest_provider = active_manifest_provider
        self._loaded_manifest_provider = loaded_manifest_provider
        self._model_structure_validator = model_structure_validator
        self._current_framework_provider = current_framework_provider
        self._loaded_framework_provider = loaded_framework_provider or current_framework_provider
        self._sessions_provider = sessions_provider
        self._checkpoint_names_provider = checkpoint_names_provider
        self._last_model_load_issue_provider = last_model_load_issue_provider

    @staticmethod
    def get_manifest_folder_path(manifest: object | None) -> str | None:
        """Return the manifest folder path exposed by this runtime status service."""
        if manifest is None:
            return None
        cfg = getattr(manifest, "get_configuration", lambda: None)()
        return str(getattr(cfg, "model_folder_path", "") or "") or None

    @classmethod
    def get_manifest_folder_name(cls, manifest: object | None) -> str | None:
        """Return the manifest folder name exposed by this runtime status service."""
        path = cls.get_manifest_folder_path(manifest)
        if not path:
            return None
        return os.path.basename(path.rstrip(os.sep)) or None

    @staticmethod
    def _normalize_framework_uuid(framework_uuid: str | None) -> str:
        """Normalize the framework uuid accepted by this runtime status service."""
        return str(framework_uuid or "").strip().lower()

    def session_has_checkpoints(
        self,
        session_name: str | None,
        framework_uuid: str | None = None,
    ) -> bool:
        """Return the session has checkpoints value owned by this runtime status service."""
        if not session_name:
            return False
        return bool(self._checkpoint_names_provider(session_name, framework_uuid))

    def get_latest_session_for_current_model(self) -> str | None:
        """Return the latest session for current model exposed by this runtime status service."""
        sessions = self._sessions_provider(self._current_framework_provider())
        return sessions[0] if sessions else None

    def get_latest_checkpoint_for_session(self, session_name: str) -> str | None:
        """Return the latest checkpoint for session exposed by this runtime status service."""
        framework_uuid = self._current_framework_provider()
        checkpoints = self._checkpoint_names_provider(session_name, framework_uuid)
        return checkpoints[0] if checkpoints else None

    def has_loaded_runtime(self) -> bool:
        """Return whether an imported model runtime currently owns the active slot."""

        return self._loaded_manifest_provider() is not None

    def get_last_model_load_issue(self) -> str | None:
        """Return the latest request-bound model loading issue, if any."""

        return self._last_model_load_issue_provider()

    def is_same_loaded_model(
        self,
        model_manifest: object | None,
        framework_uuid: str | None = None,
    ) -> bool:
        """Return whether same loaded model applies to this runtime status service."""
        loaded_manifest = self._loaded_manifest_provider()
        if loaded_manifest is None or model_manifest is None:
            return False
        if self.get_manifest_folder_path(loaded_manifest) != self.get_manifest_folder_path(
            model_manifest
        ):
            return False
        if framework_uuid is None:
            return True
        return self._normalize_framework_uuid(
            self._loaded_framework_provider()
        ) == self._normalize_framework_uuid(framework_uuid)

    def current_model_has_valid_structure(self) -> bool:
        """Validate the active manifest without exposing its storage implementation."""

        active_manifest = self._active_manifest_provider()
        if active_manifest is None:
            return False
        return self._model_structure_validator(active_manifest)

    def current_model_has_valid_runtime(self, framework_uuid: str | None = None) -> bool:
        """Return the current model has valid runtime value owned by this runtime status service."""
        if self._active_manifest_provider() is None:
            return False

        effective_framework_uuid = framework_uuid or self._current_framework_provider()
        for session_name in self._sessions_provider(effective_framework_uuid):
            if self._checkpoint_names_provider(session_name, effective_framework_uuid):
                return True

        return False

    def current_model_can_evaluate(self, framework_uuid: str | None = None) -> bool:
        """Return the current model can evaluate value owned by this runtime status service."""
        return self.current_model_has_valid_runtime(framework_uuid=framework_uuid)

    def current_model_can_infer(self, framework_uuid: str | None = None) -> bool:
        """Report whether at least one compatible checkpoint can be selected."""

        return self.current_model_has_valid_runtime(framework_uuid=framework_uuid)

    def current_model_has_training_session(self) -> bool:
        """Return the current model has training session value owned by this runtime status
        service.
        """
        return self.get_latest_session_for_current_model() is not None

    def get_current_model_runtime_capabilities(self) -> RuntimeCapabilities:
        """Return the current model runtime capabilities exposed by this runtime status service."""
        has_active_manifest = self._active_manifest_provider() is not None
        framework_uuid = self._current_framework_provider()
        sessions = self._sessions_provider(framework_uuid)
        has_inference_runtime = bool(
            has_active_manifest
            and any(
                self._checkpoint_names_provider(session_name, framework_uuid)
                for session_name in sessions
            )
        )
        return RuntimeCapabilities(
            has_training_session=bool(sessions),
            has_evaluation_runtime=bool(
                has_active_manifest and (sessions or has_inference_runtime)
            ),
            has_inference_runtime=has_inference_runtime,
        )

    def get_model_ready_presentation(self) -> ModelReadyPresentation:
        """Return the model ready presentation exposed by this runtime status service."""
        capabilities = self.get_current_model_runtime_capabilities()
        if capabilities.has_training_session and capabilities.has_evaluation_runtime:
            return ModelReadyPresentation(
                activity="Model ready",
                step="Training + Evaluation available",
                training_only_warning=False,
            )
        return ModelReadyPresentation(
            activity="Model ready",
            step="Training-only (missing session/checkpoint runtime)",
            training_only_warning=True,
        )

    def build_run_mode_execution_plan(
        self,
        mode: RLModelRunningModes,
    ) -> RunModeExecutionPlan:
        """Build the run mode execution plan required by this runtime status service."""
        framework_uuid = self._current_framework_provider()
        if mode == RLModelRunningModes.EVALUATION:
            return RunModeExecutionPlan(
                mode=RLModelRunningModes.EVALUATION,
                refresh_tensorboard_writer=True,
                start_tensorboard=True,
            )

        if mode == RLModelRunningModes.INFERENCE:
            return RunModeExecutionPlan(
                mode=RLModelRunningModes.INFERENCE,
                refresh_tensorboard_writer=False,
                start_tensorboard=False,
            )

        return RunModeExecutionPlan(
            mode=RLModelRunningModes.TRAINING,
            refresh_tensorboard_writer=(str(framework_uuid or "").strip().lower() != "sb3"),
            start_tensorboard=True,
        )
