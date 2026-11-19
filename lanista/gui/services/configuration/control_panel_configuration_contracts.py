# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed contracts shared by the configuration dialog and ControlPanel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from lanista.gui.contracts import UiNotification
from lanista.reinforcengine.rl_model.evaluation.contracts import LibraryEvaluationScope
from lanista.reinforcengine.rl_model.execution_contracts import (
    EvaluationPacing,
    EvaluationVisualization,
)
from lanista.reinforcengine.rl_model.model import RLModelRunningModes

EvaluationTarget = Literal["single", "library"]


class ConfigurationDialogAction(Enum):
    """Actions emitted by the run-configuration dialog."""

    REFRESH = "refresh"
    REFRESH_SESSION_SELECTOR = "refresh_session_selector"
    REFRESH_CHECKPOINT_SELECTOR = "refresh_checkpoint_selector"
    APPLY = "primary"
    LOAD_MODEL = "load_model"
    INSPECT_TENSORBOARD = "inspect_tensorboard"
    CREATE_COMPARISON_PACK = "create_comparison_pack"
    EDIT_MODEL = "edit_details"
    EDIT_SESSION = "edit_session_details"
    EDIT_CHECKPOINT = "edit_checkpoint_details"
    CREATE_MODEL = "new_model"
    CREATE_SESSION = "new_session"


class ConfigurationCommand(Enum):
    """Commands the ControlPanel adapter may send to HomeScreen."""

    NONE = "none"
    APPLY_CONFIGURATION = "apply_configuration"
    LOAD_MODEL = "load_model"
    INSPECT_TENSORBOARD = "inspect_tensorboard"
    CREATE_COMPARISON_PACK = "create_comparison_pack"
    EDIT_MODEL = "edit_details"
    EDIT_SESSION = "edit_session_details"
    EDIT_CHECKPOINT = "edit_checkpoint_details"
    CREATE_MODEL = "new_model"
    CREATE_SESSION = "new_session"


class ConfigurationContinuation(Enum):
    """Continuation policy after the current modal closes."""

    CLOSE = "close"
    REOPEN = "reopen"
    REOPEN_SESSION_SELECTOR = "reopen_session_selector"
    REOPEN_CHECKPOINT_SELECTOR = "reopen_checkpoint_selector"
    REOPEN_AFTER_COMMAND = "reopen_after_command"


@dataclass(frozen=True, slots=True)
class ConfigurationDialogSelection:
    """Presence-aware selection captured when a dialog result is emitted."""

    mode: RLModelRunningModes
    framework_uuid: str
    session_name: str | None
    checkpoint_name: str | None
    expected_model_uuid: str | None = None
    evaluation_visualization: EvaluationVisualization = "gui"
    evaluation_pacing: EvaluationPacing = "maximum-throughput"
    evaluation_scenario_index: int | None = None
    evaluation_library_scope: LibraryEvaluationScope | None = None
    evaluation_target: EvaluationTarget = "single"

    def __post_init__(self) -> None:
        if self.evaluation_scenario_index is not None and self.evaluation_library_scope is not None:
            raise ValueError("A single scenario and a library cannot be selected together.")


@dataclass(frozen=True, slots=True)
class ConfigurationDialogResult:
    """Immutable result returned by the run-configuration modal."""

    action: ConfigurationDialogAction
    mode: RLModelRunningModes
    framework_uuid: str
    session_name: str | None
    checkpoint_name: str | None
    expected_model_uuid: str | None = None
    evaluation_visualization: EvaluationVisualization = "gui"
    evaluation_pacing: EvaluationPacing = "maximum-throughput"
    evaluation_scenario_index: int | None = None
    evaluation_library_scope: LibraryEvaluationScope | None = None
    evaluation_target: EvaluationTarget = "single"

    def __post_init__(self) -> None:
        if self.evaluation_scenario_index is not None and self.evaluation_library_scope is not None:
            raise ValueError("A single scenario and a library cannot be selected together.")

    @property
    def selection(self) -> ConfigurationDialogSelection:
        """Return the selection exposed by this configuration dialog result."""
        return ConfigurationDialogSelection(
            mode=self.mode,
            framework_uuid=self.framework_uuid,
            session_name=self.session_name,
            checkpoint_name=self.checkpoint_name,
            expected_model_uuid=self.expected_model_uuid,
            evaluation_visualization=self.evaluation_visualization,
            evaluation_pacing=self.evaluation_pacing,
            evaluation_scenario_index=self.evaluation_scenario_index,
            evaluation_library_scope=self.evaluation_library_scope,
            evaluation_target=self.evaluation_target,
        )


@dataclass(frozen=True, slots=True)
class ControlPanelConfigurationDialogPlan:
    """Pure routing decision consumed by the Textual adapter."""

    selection: ConfigurationDialogSelection | None = None
    command: ConfigurationCommand = ConfigurationCommand.NONE
    continuation: ConfigurationContinuation = ConfigurationContinuation.CLOSE
    blocked_notification: UiNotification | None = None

    @property
    def commit_runtime_selection(self) -> bool:
        """Return the commit runtime selection exposed by this control panel configuration dialog
        plan.
        """
        return self.command is ConfigurationCommand.APPLY_CONFIGURATION

    @property
    def reopen_dialog(self) -> bool:
        """Return the reopen dialog exposed by this control panel configuration dialog plan."""
        return self.continuation in {
            ConfigurationContinuation.REOPEN,
            ConfigurationContinuation.REOPEN_SESSION_SELECTOR,
            ConfigurationContinuation.REOPEN_CHECKPOINT_SELECTOR,
        }

    @property
    def auto_open_session_selector(self) -> bool:
        """Return the auto open session selector exposed by this control panel configuration dialog
        plan.
        """
        return self.continuation is ConfigurationContinuation.REOPEN_SESSION_SELECTOR

    @property
    def auto_open_checkpoint_selector(self) -> bool:
        """Return the auto open checkpoint selector exposed by this control panel configuration
        dialog plan.
        """
        return self.continuation is ConfigurationContinuation.REOPEN_CHECKPOINT_SELECTOR

    @property
    def reopen_startup_dialog(self) -> bool:
        """Return the reopen startup dialog exposed by this control panel configuration dialog
        plan.
        """
        return self.continuation is ConfigurationContinuation.REOPEN_AFTER_COMMAND
