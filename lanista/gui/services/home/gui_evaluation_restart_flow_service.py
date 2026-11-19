# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Pure restart-intent decisions after an embedded GUI evaluation."""

from __future__ import annotations

from lanista.application_restart_service import ApplicationRestartRequest
from lanista.reinforcengine.application_errors import ApplicationValidationError
from lanista.reinforcengine.rl_model.evaluation_restart_contracts import (
    GuiEvaluationRestartContinuation,
    GuiEvaluationRouteRestartContinuation,
)


class GuiEvaluationRestartFlowService:
    """Build validated replay or navigation process-restart requests."""

    def build_request(
        self,
        destination: str,
        replay: GuiEvaluationRestartContinuation | None,
    ) -> ApplicationRestartRequest:
        """Build one validated request for the chosen post-run action."""

        if destination == "run-again":
            if replay is None:
                raise ApplicationValidationError(
                    "No completed GUI evaluation can be restored."
                )
            return replay.to_restart_request()
        return GuiEvaluationRouteRestartContinuation(
            destination  # type: ignore[arg-type]
        ).to_restart_request()


__all__ = ["GuiEvaluationRestartFlowService"]
