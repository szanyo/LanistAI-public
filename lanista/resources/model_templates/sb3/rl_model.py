# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Define the data model for rl."""

from __future__ import annotations

from typing import Never

from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_technologies.runtime_contracts import (
    CallableEnvironmentFactory,
    FrameworkLogic,
    FrameworkLogicLoggerContext,
)


def _build_environment(
    _runtime: BaseModelRuntime,
    _context: FrameworkLogicLoggerContext,
) -> FrameworkLogic:
    """Build the environment required by this rl model module."""
    raise NotImplementedError(
        "Implement the SB3 environment factory in resources/sb3/rl_model.py."
    )


class TemplateModel(BaseModelRuntime):
    """Generic framework template model."""

    framework_uuid = "sb3"
    environment_factory = CallableEnvironmentFactory(_build_environment)

    async def initialize_async(self) -> TemplateModel:
        """Initialize the async owned by this template model."""
        return self

    def _train(self) -> Never:
        """Train the operation owned by this template model."""
        self._update_progress(
            status=RLModelState.FAILED,
            additional_status_message="Template model is not implemented yet.",
        )
        raise NotImplementedError("Implement training in resources/sb3/rl_model.py.")

    def _evaluate(self) -> Never:
        """Evaluate the operation owned by this template model."""
        self._update_progress(
            status=RLModelState.FAILED,
            additional_status_message="Template model is not implemented yet.",
        )
        raise NotImplementedError("Implement evaluation in resources/sb3/rl_model.py.")
