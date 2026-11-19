# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate control panel configuration flow."""

from __future__ import annotations

from typing import Protocol

from lanista.gui.contracts import UiNotification, normalize_ui_severity
from lanista.gui.services.configuration.control_panel_configuration_contracts import (
    ConfigurationCommand,
    ConfigurationContinuation,
    ConfigurationDialogAction,
    ConfigurationDialogResult,
    ControlPanelConfigurationDialogPlan,
)
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.run_precheck_service import ActionPrecheck


class CreateSessionPrecheckPort(Protocol):
    """Narrow application port needed by configuration routing."""

    def evaluate_create_session(
        self,
        mode: RLModelRunningModes,
    ) -> ActionPrecheck:
        """Evaluate the create session owned by this create session precheck port."""
        ...


_ACTION_PLANS = {
    ConfigurationDialogAction.REFRESH: (
        ConfigurationCommand.NONE,
        ConfigurationContinuation.REOPEN,
    ),
    ConfigurationDialogAction.REFRESH_SESSION_SELECTOR: (
        ConfigurationCommand.NONE,
        ConfigurationContinuation.REOPEN_SESSION_SELECTOR,
    ),
    ConfigurationDialogAction.REFRESH_CHECKPOINT_SELECTOR: (
        ConfigurationCommand.NONE,
        ConfigurationContinuation.REOPEN_CHECKPOINT_SELECTOR,
    ),
    ConfigurationDialogAction.APPLY: (
        ConfigurationCommand.APPLY_CONFIGURATION,
        ConfigurationContinuation.CLOSE,
    ),
    ConfigurationDialogAction.LOAD_MODEL: (
        ConfigurationCommand.LOAD_MODEL,
        ConfigurationContinuation.REOPEN_AFTER_COMMAND,
    ),
    ConfigurationDialogAction.INSPECT_TENSORBOARD: (
        ConfigurationCommand.INSPECT_TENSORBOARD,
        ConfigurationContinuation.REOPEN_AFTER_COMMAND,
    ),
    ConfigurationDialogAction.CREATE_COMPARISON_PACK: (
        ConfigurationCommand.CREATE_COMPARISON_PACK,
        ConfigurationContinuation.CLOSE,
    ),
    ConfigurationDialogAction.EDIT_MODEL: (
        ConfigurationCommand.EDIT_MODEL,
        ConfigurationContinuation.CLOSE,
    ),
    ConfigurationDialogAction.EDIT_SESSION: (
        ConfigurationCommand.EDIT_SESSION,
        ConfigurationContinuation.CLOSE,
    ),
    ConfigurationDialogAction.EDIT_CHECKPOINT: (
        ConfigurationCommand.EDIT_CHECKPOINT,
        ConfigurationContinuation.CLOSE,
    ),
    ConfigurationDialogAction.CREATE_MODEL: (
        ConfigurationCommand.CREATE_MODEL,
        ConfigurationContinuation.REOPEN_AFTER_COMMAND,
    ),
}


class ControlPanelConfigurationFlowService:
    """Decision helpers for ControlPanel configuration dialog routing."""

    def __init__(self, precheck_service: CreateSessionPrecheckPort) -> None:
        """Initialize the control panel configuration flow service and its owned state."""
        self._precheck_service = precheck_service

    def build_result_plan(
        self,
        result: ConfigurationDialogResult | None,
    ) -> ControlPanelConfigurationDialogPlan:
        """Build the result plan required by this control panel configuration flow service."""
        if result is None:
            return ControlPanelConfigurationDialogPlan()

        if result.action is ConfigurationDialogAction.CREATE_SESSION:
            precheck = self._precheck_service.evaluate_create_session(result.mode)
            if not precheck.allowed:
                return ControlPanelConfigurationDialogPlan(
                    selection=result.selection,
                    blocked_notification=UiNotification(
                        message=precheck.message or "Action blocked.",
                        title=precheck.title,
                        severity=normalize_ui_severity(precheck.severity),
                    )
                )
            return ControlPanelConfigurationDialogPlan(
                selection=result.selection,
                command=ConfigurationCommand.CREATE_SESSION,
                continuation=ConfigurationContinuation.REOPEN_AFTER_COMMAND,
            )

        command, continuation = _ACTION_PLANS[result.action]
        return ControlPanelConfigurationDialogPlan(
            selection=result.selection,
            command=command,
            continuation=continuation,
        )
