# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate inspect toggle."""

from dataclasses import dataclass
from typing import Callable, Optional

from lanista.reinforcengine.rl_model.model import RLModelRunningModes


@dataclass
class InspectTogglePrecheck:
    """Carry the data required for inspect toggle precheck."""

    action: str  # "stop" | "start" | "blocked"
    message: str
    title: str
    severity: str


@dataclass
class InspectToggleExecutionPlan:
    """Describe the executable plan for inspect toggle execution."""

    action: str  # "stop" | "start" | "blocked"
    message: str
    title: str
    severity: str
    restart_context: Optional[object] = None


class InspectToggleService:
    """Coordinate inspect toggle operations."""

    def __init__(
        self,
        *,
        has_current_model_provider: Callable[[], bool],
        restart_context_builder: Callable[[RLModelRunningModes, Optional[str]], Optional[object]],
    ) -> None:
        """Initialize the inspect toggle service and its owned state."""
        self._has_current_model_provider = has_current_model_provider
        self._restart_context_builder = restart_context_builder

    def evaluate_preconditions(self, *, inspect_is_running: bool) -> InspectTogglePrecheck:
        """Evaluate the preconditions owned by this inspect toggle service."""
        if inspect_is_running:
            return InspectTogglePrecheck(
                action="stop",
                message="Inspect mode stopped.",
                title="Inspect",
                severity="information",
            )

        if not self._has_current_model_provider():
            return InspectTogglePrecheck(
                action="blocked",
                message="Inspect requires a loaded model.",
                title="Inspect blocked",
                severity="warning",
            )

        return InspectTogglePrecheck(
            action="start",
            message="Inspect mode active.",
            title="Inspect",
            severity="information",
        )

    def build_execution_plan(
        self,
        *,
        inspect_is_running: bool,
        selected_session: Optional[str],
    ) -> InspectToggleExecutionPlan:
        """Build the execution plan required by this inspect toggle service."""
        precheck = self.evaluate_preconditions(inspect_is_running=inspect_is_running)

        if precheck.action != "start":
            return InspectToggleExecutionPlan(
                action=precheck.action,
                message=precheck.message,
                title=precheck.title,
                severity=precheck.severity,
                restart_context=None,
            )

        restart_context = self._restart_context_builder(
            RLModelRunningModes.INFERENCE,
            selected_session,
        )
        if restart_context is None:
            return InspectToggleExecutionPlan(
                action="blocked",
                message="Inspect requires a loaded model framework context.",
                title="Inspect blocked",
                severity="warning",
                restart_context=None,
            )

        return InspectToggleExecutionPlan(
            action=precheck.action,
            message=precheck.message,
            title=precheck.title,
            severity=precheck.severity,
            restart_context=restart_context,
        )
