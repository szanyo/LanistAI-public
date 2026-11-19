# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Transport-neutral GUI evaluation context restored across app restarts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from lanista.application_restart_service import ApplicationRestartRequest
from lanista.reinforcengine.application_errors import ApplicationValidationError
from lanista.reinforcengine.rl_model.execution_contracts import EvaluationPacing

GUI_EVALUATION_RESTART_REASON = "gui-evaluation-native-reset"
GUI_EVALUATION_ROUTE_RESTART_REASON = "gui-evaluation-navigation-reset"
GuiEvaluationRestartDestination = Literal[
    "configuration",
    "load-model",
    "session",
    "checkpoint",
    "mode",
]
_GUI_EVALUATION_RESTART_DESTINATIONS = frozenset(
    {"configuration", "load-model", "session", "checkpoint", "mode"}
)


@dataclass(frozen=True, slots=True)
class GuiEvaluationRestartContinuation:
    """Runtime identity and fixed scenario restored after process replacement."""

    model_uuid: str
    framework_uuid: str
    session_name: str
    checkpoint_name: str | None
    pacing: EvaluationPacing
    scenario_index: int

    def to_restart_request(self) -> ApplicationRestartRequest:
        """Convert this gui evaluation restart continuation's value to restart request."""
        return ApplicationRestartRequest(
            reason=GUI_EVALUATION_RESTART_REASON,
            continuation={
                "model_uuid": self.model_uuid,
                "framework_uuid": self.framework_uuid,
                "session_name": self.session_name,
                "checkpoint_name": self.checkpoint_name,
                "pacing": self.pacing,
                "scenario_index": self.scenario_index,
            },
        )

    @classmethod
    def from_restart_request(
        cls,
        request: ApplicationRestartRequest | None,
    ) -> GuiEvaluationRestartContinuation | None:
        """Create this gui evaluation restart continuation's value from restart request."""
        if request is None or request.reason != GUI_EVALUATION_RESTART_REASON:
            return None
        payload = request.continuation
        try:
            model_uuid = cls._required_text(payload, "model_uuid")
            framework_uuid = cls._required_text(payload, "framework_uuid").lower()
            session_name = cls._required_text(payload, "session_name")
            checkpoint_value = payload.get("checkpoint_name")
            checkpoint_name = (
                str(checkpoint_value).strip() if checkpoint_value is not None else None
            )
            pacing_value = cls._required_text(payload, "pacing").lower()
            scenario_index = int(payload["scenario_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ApplicationValidationError(
                "The GUI evaluation restart continuation is invalid."
            ) from error
        if pacing_value not in {"maximum-throughput", "realtime"}:
            raise ApplicationValidationError(
                "The GUI evaluation restart pacing is invalid."
            )
        if scenario_index < 0:
            raise ApplicationValidationError(
                "The GUI evaluation restart scenario index is invalid."
            )
        return cls(
            model_uuid=model_uuid,
            framework_uuid=framework_uuid,
            session_name=session_name,
            checkpoint_name=checkpoint_name or None,
            pacing=pacing_value,  # type: ignore[arg-type]
            scenario_index=scenario_index,
        )

    @staticmethod
    def _required_text(payload: dict[str, object], name: str) -> str:
        """Return the required text value owned by this gui evaluation restart continuation."""
        value = str(payload.get(name, "") or "").strip()
        if not value:
            raise ValueError(f"Missing restart continuation field: {name}.")
        return value


@dataclass(frozen=True, slots=True)
class GuiEvaluationRouteRestartContinuation:
    """Allowlisted Home-screen destination restored after process replacement."""

    destination: GuiEvaluationRestartDestination

    def to_restart_request(self) -> ApplicationRestartRequest:
        """Convert this validated destination into a restart request."""

        self._validate_destination(self.destination)
        return ApplicationRestartRequest(
            reason=GUI_EVALUATION_ROUTE_RESTART_REASON,
            continuation={"destination": self.destination},
        )

    @classmethod
    def from_restart_request(
        cls,
        request: ApplicationRestartRequest | None,
    ) -> GuiEvaluationRouteRestartContinuation | None:
        """Decode one allowlisted destination from a restart request."""

        if request is None or request.reason != GUI_EVALUATION_ROUTE_RESTART_REASON:
            return None
        destination = str(request.continuation.get("destination", "") or "").strip()
        cls._validate_destination(destination)
        return cls(cast(GuiEvaluationRestartDestination, destination))

    @staticmethod
    def _validate_destination(destination: str) -> None:
        """Reject unrecognized navigation names before process replacement."""

        if destination not in _GUI_EVALUATION_RESTART_DESTINATIONS:
            raise ApplicationValidationError(
                "The GUI evaluation restart destination is invalid."
            )


__all__ = [
    "GUI_EVALUATION_RESTART_REASON",
    "GUI_EVALUATION_ROUTE_RESTART_REASON",
    "GuiEvaluationRestartContinuation",
    "GuiEvaluationRestartDestination",
    "GuiEvaluationRouteRestartContinuation",
]
